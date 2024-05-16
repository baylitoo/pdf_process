import glob
import pandas as pd
import numpy as np
import os
from datasets.features import ClassLabel
from transformers import AutoProcessor
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import LiltForTokenClassification, LayoutLMv3FeatureExtractor, AutoTokenizer, LayoutLMv3Processor
from pdf2image import convert_from_bytes
from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial.distance import cdist
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
import cv2
import pdfplumber
from datasets import load_metric

class PDFProcessor:
    def __init__(self, model_path, poppler_path):
        self.model = LiltForTokenClassification.from_pretrained(model_path)
        self.model_id = "SCUT-DLVCLab/lilt-roberta-en-base"
        self.feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.processor = LayoutLMv3Processor(self.feature_extractor, self.tokenizer)
        self.poppler_path = poppler_path
        self.metric = load_metric("seqeval")
        self.label2color = {
            'B-SECTION-HEADER': "blue", 'E-SECTION-HEADER': "blue", 'I-SECTION-HEADER': "blue", 'S-SECTION-HEADER': "blue",
            'B-TEXT': "orange", 'E-TEXT': "orange", 'I-TEXT': "orange", 'S-TEXT': "orange",
            'B-TITLE': "green", 'E-TITLE': "green", 'I-TITLE': "green", 'S-TITLE': "green",
            "O": "red"
        }

    def unnormalize_box(self, bbox, width, height):
        return [
            width * (bbox[0] / 1000),
            height * (bbox[1] / 1000),
            width * (bbox[2] / 1000),
            height * (bbox[3] / 1000),
        ]

    def draw_boxes(self, image, boxes, predictions):
        width, height = image.size
        normalizes_boxes = [self.unnormalize_box(box, width, height) for box in boxes]

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for prediction, box in zip(predictions, normalizes_boxes):
            if prediction == "O":
                continue
            draw.rectangle(box, outline="black")
            draw.rectangle(box, outline=self.label2color[prediction])
            draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=self.label2color[prediction], font=font)
        return image

    def chunks(self, tensor, chunk_size):
        num_chunks = (tensor.size(0) + chunk_size - 1) // chunk_size
        for i in range(num_chunks):
            yield tensor[i * chunk_size:(i + 1) * chunk_size]

    def run_inference(self, image, output_image=True):
        encoding = self.processor(image, return_tensors="pt")
        del encoding["pixel_values"]

        input_id_chunks = list(self.chunks(encoding["input_ids"][0], 512))
        bbox_chunks = list(self.chunks(encoding["bbox"][0], 512))

        all_labels = []
        all_bboxes = []
        all_texts = []
        for input_ids, bboxes in zip(input_id_chunks, bbox_chunks):
            chunk_encoding = {"input_ids": input_ids.unsqueeze(0), "bbox": bboxes.unsqueeze(0)}

            outputs = self.model(**chunk_encoding)
            predictions = outputs.logits.argmax(-1).squeeze()

            if predictions.dim() == 0:
                predictions = [predictions.item()]
            else:
                predictions = predictions.tolist()

            labels = [self.model.config.id2label[prediction] for prediction in predictions]
            all_labels.extend(labels)
            bboxes = bboxes.tolist()
            all_bboxes.extend(bboxes)

            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            all_texts.extend(texts)

        df = pd.DataFrame({
            'bbox': all_bboxes,
            'text': all_texts,
            'label': all_labels
        })
        if output_image:
            image = self.draw_boxes(image, encoding["bbox"][0], all_labels)
            return df, image

        return df

    def convert_pdf_to_images(self, pdf_file_path):
        with open(pdf_file_path, "rb") as f:
            pdf_file_bytes = f.read()
        images = convert_from_bytes(pdf_file_bytes, poppler_path=self.poppler_path)

        if len(images) > 1:
            print(f"Le fichier PDF contient {len(images)} pages.")

        image_files = []
        for i, image in enumerate(images):
            img_file_path = f"pdf_page_{i}.png"
            image.save(img_file_path, "PNG")
            image_files.append(img_file_path)

        return images, image_files

    def label_func(self, x):
        if x in ['B-SECTION-HEADER', 'E-SECTION-HEADER', 'I-SECTION-HEADER', 'S-SECTION-HEADER']:
            return 10
        elif x in ['B-TEXT', 'E-TEXT', 'I-TEXT', 'S-TEXT']:
            return 3
        elif x in ['B-TITLE', 'E-TITLE', 'I-TITLE', 'S-TITLE']:
            return 2
        else:
            print(f"Unrecognized label: {x}")
            return 0

    def process_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        df, img = self.run_inference(image, output_image=True)

        dff = df.copy()
        dff['label'] = dff['label'].apply(self.label_func)
        dff[['x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright']] = pd.DataFrame(dff['bbox'].to_list())

        points = dff[['x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright', 'label']].values

        df_withoutboxes = dff.drop("bbox", axis=1)

        x_threshold = 50
        grouped_df = pd.DataFrame(columns=dff.columns)
        dff = dff.sort_values(by=['y_topleft', 'x_topleft'])

        for _, word in dff.iterrows():
            if len(grouped_df) > 0 and abs(word['x_topleft'] - grouped_df.iloc[-1]['x_topleft']) < x_threshold and word['label'] == grouped_df.iloc[-1]['label']:
                grouped_df.loc[grouped_df.index[-1], 'x_bottomright'] = max(grouped_df.iloc[-1]['x_bottomright'], word['x_bottomright'])
                grouped_df.loc[grouped_df.index[-1], 'y_bottomright'] = max(grouped_df.iloc[-1]['y_bottomright'], word['y_bottomright'])
                grouped_df.loc[grouped_df.index[-1], 'text'] += ' ' + word['text']
            else:
                grouped_df = pd.concat([grouped_df, pd.DataFrame(word).T])

        df_withoutboxes = grouped_df.drop("bbox", axis=1)
        pointss = df_withoutboxes[['x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright', 'label']].values

        pointss = np.array(pointss).astype(float)
        distances = cdist(pointss, pointss, 'euclidean')

        vertical_weight = 2
        for threshold in [100]:
            for i in range(len(pointss)):
                for j in range(len(pointss)):
                    x1, y1 = pointss[i][:2]
                    x2, y2 = pointss[j][:2]
                    vertical_dist = abs(y2 - y1)
                    horizontal_dist = abs(x2 - x1)
                    distances[i, j] = np.sqrt((vertical_dist * vertical_weight) ** 2 + horizontal_dist ** 2)

        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(distances < threshold, np.where(distances == 0, 0, 1 / distances), 0)

        for i in range(len(pointss)):
            for j in range(len(pointss)):
                if (pointss[i][4] == 10 or pointss[i][4] == 2) and (pointss[i][1] < pointss[j][1]) and (pointss[j][4] not in [10, 2]):
                    weights[i, j] *= 5

        G = nx.from_numpy_array(weights)
        best_modularity = -np.inf
        best_resolution = None
        for resolution in np.linspace(1, 2, 10):
            partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
            if len(G.edges()) > 0:
                modularity = community_louvain.modularity(partition, G, weight='weight')
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_resolution = resolution

        partition = community_louvain.best_partition(G, weight='weight', resolution=best_resolution)
        pos = {i: (pointss[i][0], -pointss[i][1]) for i in range(len(pointss))}
        colors = [partition[i] for i in range(len(pointss))]
        nx.draw(G, pos, node_color=colors, with_labels=False)
        plt.show()
        print("viva l'algèrie")

        image_width = np.shape(img)[1]
        image_height = np.shape(img)[0]

        def normalize_box(bbox, width, height):
            return [
                (bbox["x_topleft"] * width) / 1000,
                (bbox["y_topleft"] * height) / 1000,
                (bbox["x_bottomright"] * width) / 1000,
                (bbox["y_bottomright"] * height) / 1000,
            ]

        img = cv2.imread(img_path)
        thresholdd = 3
        image_width = np.shape(img)[1]
        image_height = np.shape(img)[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(set(partition.values()))))
        pos_array = np.array(list(pos.values()))
        big_boxes = []

        for cluster, color in zip(set(partition.values()), colors):
            points = pos_array[[i for i, c in enumerate(partition.values()) if c == cluster]]
            indices = [i for i, c in enumerate(partition.values()) if c == cluster]
            if len(points) < thresholdd:
                continue
            boxes = df_withoutboxes.iloc[indices]
            boxess = [normalize_box(box.to_dict(), image_width, image_height) for _, box in boxes.iterrows()]
            boxess_df = pd.DataFrame(boxess, columns=['x_topleft', 'y_topleft', 'x_bottomright', 'y_bottomright'])
            x_min = boxess_df['x_topleft'].min()
            y_min = boxess_df['y_topleft'].min()
            x_max = boxess_df['x_bottomright'].max()
            y_max = boxess_df['y_bottomright'].max()
            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            big_boxes.append(((x_min, y_min), (x_max, y_max)))

        cv2.imshow("CV", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        x_coords = [box[0][0] for box in big_boxes]
        if len(big_boxes) > 1:
            n_clusters = 2
        else:
            n_clusters = 1
        kmeans = KMeans(n_clusters=n_clusters, n_init=10).fit(np.array(x_coords).reshape(-1, 1))
        separation_line_x = np.mean(kmeans.cluster_centers_)
        sorted_boxes = sorted(big_boxes, key=lambda box: (box[0][0] < separation_line_x, box[0][0]))

        def area_of_intersection(box1, box2):
            x_min = max(box1[0][0], box2[0][0])
            y_min = max(box1[0][1], box2[0][1])
            x_max = min(box1[1][0], box2[1][0])
            y_max = min(box1[1][1], box2[1][1])
            if x_min < x_max and y_min < y_max:
                return (x_max - x_min) * (y_max - y_min)
            else:
                return 0

        def merge_boxes_with_threshold(boxes, threshold):
            merged_boxes = boxes.copy()
            i = 0
            while i < len(merged_boxes):
                j = i + 1
                while j < len(merged_boxes):
                    if area_of_intersection(merged_boxes[i], merged_boxes[j]) > threshold:
                        x_min = min(merged_boxes[i][0][0], merged_boxes[j][0][0])
                        y_min = min(merged_boxes[i][0][1], merged_boxes[j][0][1])
                        x_max = max(merged_boxes[i][1][0], merged_boxes[j][1][0])
                        y_max = max(merged_boxes[i][1][1], merged_boxes[j][1][1])
                        merged_boxes[i] = ((x_min, y_min), (x_max, y_max))
                        del merged_boxes[j]
                        j -= 1
                    j += 1
                i += 1
            return merged_boxes

        threshold = 2000
        merged_boxes = merge_boxes_with_threshold(sorted_boxes, threshold)

        int_boxes = [((int(x1), int(y1)), (int(x2), int(y2))) for (x1, y1), (x2, y2) in merged_boxes]
        for (x1, y1), (x2, y2) in int_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("CV", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return merged_boxes

    def extract_text_from_pdf(self, pdf_file_path, merged_boxes, page_number):
        text = []
        with pdfplumber.open(pdf_file_path) as pdf:
            if 0 <= page_number < len(pdf.pages):
                page = pdf.pages[page_number]
                rois = []

                scale_x = page.width / 1000
                scale_y = page.height / 1000
                scaled_boxes = [((x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y)) for (x1, y1), (x2, y2) in merged_boxes]
                for box in scaled_boxes:
                    (x1, y1), (x2, y2) = box
                    if y1 < 8:
                        y1 = 8
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(page.width, x2), min(page.height, y2)
                    if x1 >= x2 or y1 >= y2:
                        print(f"Invalid box coordinates: ({x1}, {y1}, {x2}, {y2})")
                        continue
                    box = (x1, y1, x2, y2)
                    roi = page.crop([int(float(x)) for x in box])
                    text.append(roi.extract_text(x_tolerance=1, y_tolerance=3, layout=True, x_density=7.25, y_density=13))
                    rois.append(roi)
            else:
                print(f"Page number {page_number} out of range. Skipping...")

        return text

    def process_pdfs(self, input_dir, output_dir):
        for index_pdf, pdf_file_path in enumerate(glob.glob(f'{input_dir}/*.pdf')):
            textss = []
            print(f'Processing: {pdf_file_path}')

            images, image_files = self.convert_pdf_to_images(pdf_file_path)
            for index_img, img_path in enumerate(image_files):
                merged_boxes = self.process_image(img_path)
                text = self.extract_text_from_pdf(pdf_file_path, merged_boxes, index_img)
                textss.append(text)

            base_name = os.path.basename(pdf_file_path)
            file_name_without_extension = os.path.splitext(base_name)[0]
            file_path = os.path.join(output_dir, f"textcv_{index_pdf}_{file_name_without_extension}.txt")
            with open(file_path, 'w', encoding='utf-8') as file:
                for i, string_list in enumerate(textss):
                    for j, string in enumerate(string_list):
                        if string:
                            file.write(string + '\n' + f'\n paragraphe numero {i}-{j} \n\n')

if __name__ == '__main__':
    input_dir = r'C:\Users\ADBI\Documents\input_pdf'
    output_dir = r'C:\Users\ADBI\Documents\output_pdf'
    model_path = r'C:\Users\ADBI\Documents\drive-download-20230826T143041Z-001'
    poppler_path = r"C:\Users\ADBI\Documents\ANNOTATION\texte CV\poppler-23.08.0\Library\bin"

    pdf_processor = PDFProcessor(model_path, poppler_path)
    pdf_processor.process_pdfs(input_dir, output_dir)
