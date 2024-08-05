
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
from collections import Counter
import numpy as np
from io import BytesIO
import cv2
from pathlib import Path
import cv2
from yolodetect import YoloDetect
import pandas as pd
import re
import ast
import requests
import httpx

def read_image(file) -> np.ndarray:
    """Reads and decodes an image from an uploaded file."""
    # Create a BytesIO stream from the uploaded file
    image_stream = BytesIO(file)

    # Move the stream's position to the beginning
    image_stream.seek(0)

    # Read the bytes from the stream and decode them using OpenCV
    # This will decode the image data into a NumPy array
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    # Return the decoded image as a NumPy array
    return image


class DetectionOrg(BaseModel):
    """Model class to represent the results of object detection."""
    all_answers: list = None

class AutoGrade(BaseModel):
    
    # wrong_answers: int = None
    score: float = None

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Hàm sắp xếp các điểm theo thứ tự nhất định
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Hàm biến đổi phối cảnh của hình ảnh dựa trên 4 điểm đầu vào
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = distance(br, bl)
    widthB = distance(tr, tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = distance(tr, br)
    heightB = distance(tl, bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# Hàm tìm góc của hình chữ nhật xoay từ các điểm xấp xỉ
def find_corner_by_rotated_rect(box, approx):
    corner = []
    for p_box in box:
        min_dist = float('inf')
        min_p = None
        for p in approx:
            dist = distance(p_box, p[0])
            if dist < min_dist:
                min_dist = dist
                min_p = p[0]
        corner.append(min_p)
    return np.array(corner)

# Hàm xử lý hình ảnh để tìm các đáp án từ vị trí đầu đến vị trí cuối
def process_image(image_path, template_path, start_index, end_index, answers):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    rows = 4
    cols = end_index - start_index + 2  # Tính cả cột đầu tiên không sử dụng
    for pt in zip(*loc[::-1]):
        col = (pt[0] // (image.shape[1] // cols)) - 1  # Bỏ cột đầu tiên
        if col < 0:
            continue
        row = pt[1] // (image.shape[0] // rows)
        answers[start_index + col - 1] = chr(65 + row)  # 'A' = 65 trong ASCII
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return image

def final(image):
    # Đọc ảnh ban đầu và xử lý
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 2)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)

    # Tìm và vẽ đường biên cho khung trắc nghiệm
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Tìm tọa độ 4 điểm của khung
    x, y = [], []
    for a in range(2):  # Duyệt qua hai contour lớn nhất
        approx = cv2.approxPolyDP(contours[a], 0.01 * cv2.arcLength(contours[a], True), True)
        rect = cv2.minAreaRect(contours[a])
        box = cv2.boxPoints(rect).astype(np.int64)
        for i in range(len(box)):
            y.append(box[i][0])
            x.append(box[i][1])

    y_max = min(max(y) + 10, image.shape[1] - 1)
    y_min = max(min(y) - 10, 0)
    x_max = min(max(x) + 10, image.shape[0] - 1)
    x_min = max(min(x) - 10, 0)

    # Cắt vùng bảng trắc nghiệm
    img_crop = image[x_min:x_max, y_min:y_max]

    # Kiểm tra và thay đổi kích thước nếu cần
    desired_height, desired_width = 689 - 226, 1534 - 24
    current_height, current_width = img_crop.shape[:2]

    if current_height != desired_height or current_width != desired_width:
        img_crop_resized = cv2.resize(img_crop, (desired_width, desired_height))
    else:
        img_crop_resized = img_crop

    cv2.imwrite('DOAN/hinhcropx.jpg', img_crop_resized)

    # Đọc và xử lý ảnh đã crop
    gray_2 = cv2.cvtColor(img_crop_resized, cv2.COLOR_BGR2GRAY)
    blurred_2 = cv2.GaussianBlur(gray_2, (3, 3), 2)
    thresh_2 = cv2.adaptiveThreshold(blurred_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    contours, _ = cv2.findContours(thresh_2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Chuẩn bị và biến đổi các khung hình
    images_processed = []
    top_y_coordinates = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect).astype(np.int64)
        corner = find_corner_by_rotated_rect(box, approx)
        processed = four_point_transform(thresh_2, corner)
        images_processed.append(processed)
        top_y_coordinates.append(np.min(corner[:, 1]))

    if top_y_coordinates[0] < top_y_coordinates[1]:
        cv2.imwrite('DOAN/image_path_1.jpg', images_processed[0])
        cv2.imwrite('DOAN/image_path_2.jpg', images_processed[1])
    else:
        cv2.imwrite('DOAN/image_path_1.jpg', images_processed[1])
        cv2.imwrite('DOAN/image_path_2.jpg', images_processed[0])

    answers = [None] * 40
    processed_image_1 = process_image("DOAN/image_path_1.jpg", "DOAN/template.jpg", 1, 20, answers)
    processed_image_2 = process_image("DOAN/image_path_2.jpg", "DOAN/template.jpg", 21, 40, answers)

    # Chuyển đổi danh sách đáp án thành danh sách các tuple
    all_answers = [(i + 1, answers[i]) for i in range(40)]

    # Ghi nội dung vào file
    file_path = Path("DOAN/original_answers.txt")
    with file_path.open(mode='w', encoding='utf-8') as file:
        file.write(str(all_answers))

    return all_answers
def add_missing_values(ans, compare_ans):
    missing_rows = []
    for _, row in compare_ans.iterrows():
        if not ((abs(ans['left_x'] - row['left_x']) < 30).any()):
            missing_row = {"left_x": row['left_x'], "top_y": 0}
            missing_rows.append(missing_row)
    return pd.concat([ans, pd.DataFrame(missing_rows)], ignore_index=True)

def add_virtual_x(ans, xcrop):
    left_x_min, left_x_max = xcrop
    while len(ans) < 20:
        max_diff = 0
        insert_index = -1
        for i in range(len(ans) - 1):
            diff = ans.iloc[i + 1]['left_x'] - ans.iloc[i]['left_x']
            if diff > max_diff:
                max_diff = diff
                insert_index = i
        
        if max_diff > 0:
            new_x = int((ans.iloc[insert_index]['left_x'] + ans.iloc[insert_index + 1]['left_x']) / 2)
            if left_x_min <= new_x <= left_x_max:
                ans = pd.concat([ans.iloc[:insert_index + 1], pd.DataFrame([{'left_x': new_x, 'top_y': 0}]), ans.iloc[insert_index + 1:]], ignore_index=True)
            else:
                break
        else:
            break
    
    return ans

def xacdinh_da(values, max_diff=10):
    groups, current_group, zero_group, multiple_groups = [], [], [], []
    for value in values:
        if value[1] == 0:
            zero_group.append(value)
        elif value[1] == -1:
            multiple_groups.append(value)
        else:
            if not current_group or value[1] - current_group[-1][1] <= max_diff:
                current_group.append(value)
            else:
                groups.append(current_group)
                current_group = [value]
    if current_group:
        groups.append(current_group)
    return groups, zero_group, multiple_groups

# Đọc nội dung từ file
def read_answers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return ast.literal_eval(file.read().strip())
    
def part_1(image_path):
    
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)

    # Find and draw contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    x, y = [], []
    for a in range(0, 2):
        approx = cv2.approxPolyDP(contours[a], 0.01 * cv2.arcLength(contours[a], True), True)
        rect = cv2.minAreaRect(contours[a])
        box = cv2.boxPoints(rect)
        box = box.astype(np.int64)
        for i in range(len(box)):
            y.append(box[i][0])
            x.append(box[i][1])

    y_max, y_min = max(y) + 5, min(y) - 5
    x_max, x_min = max(x) + 5, min(x) - 5

    # Ensure coordinates are within image dimensions
    y_max, y_min = min(y_max, image.shape[1] - 1), max(y_min, 0)
    x_max, x_min = min(x_max, image.shape[0] - 1), max(x_min, 0)

    # Crop image
    img_crop = image[x_min:x_max, y_min:y_max]

    # Resize to desired dimensions if needed
    desired_x_min, desired_x_max = 226, 689
    desired_y_min, desired_y_max = 24, 1534
    current_height, current_width = img_crop.shape[:2]
    desired_height = desired_x_max - desired_x_min
    desired_width = desired_y_max - desired_y_min

    if current_height != desired_height or current_width != desired_width:
        img_crop_resized = cv2.resize(img_crop, (desired_width, desired_height))
    else:
        img_crop_resized = img_crop
    
    gray_2 = cv2.cvtColor(img_crop_resized, cv2.COLOR_BGR2GRAY)
    blurred_2 = cv2.GaussianBlur(gray_2, (5, 5), 0)
    thresh_2 = cv2.adaptiveThreshold(blurred_2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 3)
    contours_2, _ = cv2.findContours(thresh_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_2 = sorted(contours_2, key=lambda x: cv2.contourArea(x), reverse=True)

    top_y_1, bot_y_1 = float('inf'), 0
    top_y_2, bot_y_2 = float('inf'), 0
    left_x_1, right_x_1 = float('inf'), 0
    left_x_2, right_x_2 = float('inf'), 0
    # Get the top and bottom y coordinates for the first contour
    for contour in contours_2[0]:
        y = contour[0][1]
        x = contour[0][0]
        if y < top_y_1:
            top_y_1 = y
        elif y > bot_y_1:
            bot_y_1 = y
        if x < left_x_1:
            left_x_1 = x
        elif x > right_x_1:
            right_x_1 = x
    # Get the top and bottom y coordinates for the second contour
    for contour in contours_2[1]:
        y = contour[0][1]
        x = contour[0][0]
        if y < top_y_2:
            top_y_2 = y
        elif y > bot_y_2:
            bot_y_2 = y
        if x < left_x_2:
            left_x_2 = x
        elif x > right_x_2:
            right_x_2 = x

    # Compare and assign to crop variables
    if top_y_1 < top_y_2:
        ycrop1 = (top_y_1, bot_y_1)
        xcrop1 = (left_x_1 +10, right_x_1)
        ycrop2 = (top_y_2, bot_y_2)
        xcrop2 = (left_x_2 +10, right_x_2)
    else:
        ycrop1 = (top_y_2, bot_y_2)
        xcrop1 = (left_x_2 +10, right_x_2)
        ycrop2 = (top_y_1, bot_y_1)
        xcrop2 = (left_x_1 +10, right_x_1)
    # Lưu hình (part2)
    cv2.imwrite("img_crop.jpg", img_crop_resized)
    return xcrop1, xcrop2, ycrop1, ycrop2 

def a(xcrop1, xcrop2, ycrop1, ycrop2, output_file_path):
    with open(output_file_path, 'r') as file:
        yolo_output = file.read()

    # Biểu thức chính quy để tìm tất cả các tọa độ
    pattern = r'\[(\d+), (\d+), \d+, \d+\]'
    matches = re.findall(pattern, yolo_output)

    # Tạo danh sách các từ điển với chỉ 'left_x' và 'top_y'
    data = [{"left_x": int(match[0]), "top_y": int(match[1])} for match in matches]

    df = pd.DataFrame(data).sort_values(by='left_x').reset_index(drop=True)

    top_1, bot_1 = ycrop1
    top_2, bot_2 = ycrop2

    ans1 = df[(df['top_y'] >= top_1) & (df['top_y'] <= bot_1)]
    ans20 = df[(df['top_y'] >= top_2) & (df['top_y'] <= bot_2)]

    # So sánh 2 bảng đáp án để tạo ra đối tượng còn thiếu
    ans1 = add_missing_values(ans1, ans20)
    ans20 = add_missing_values(ans20, ans1)

    ans1 = add_virtual_x(ans1, xcrop1)
    ans20 = add_virtual_x(ans20, xcrop2)

    # Sắp xếp lại ans1 và ans20
    ans1 = ans1.sort_values(by='top_y').reset_index(drop=True)
    ans20 = ans20.sort_values(by='top_y').reset_index(drop=True)

    group1 = list(ans1.apply(lambda row: (row['left_x'], row['top_y']), axis=1))
    group20 = list(ans20.apply(lambda row: (row['left_x'], row['top_y']), axis=1))

    nhom_1, zero_group_1, multiple_groups_1 = xacdinh_da(group1)
    nhom_20, zero_group_20, multiple_groups_20 = xacdinh_da(group20)

    for group in nhom_1:
        group.sort(key=lambda x: x[0])
    for group in nhom_20:
        group.sort(key=lambda x: x[0])

    gtri_dapan = ['A', 'B', 'C', 'D', 'None','Choose_many_answers']
    nhomdau, nhomcuoi = {}, {}

    for i, group in enumerate(nhom_1):
        nhomdau[gtri_dapan[i % 4]] = group
    nhomdau['None'] = zero_group_1
    nhomdau['Choose_many_answers'] = multiple_groups_1

    for i, group in enumerate(nhom_20):
        nhomcuoi[gtri_dapan[i % 4]] = group
    nhomcuoi['None'] = zero_group_20
    nhomcuoi['Choose_many_answers'] = multiple_groups_20

    final_1, final_20 = [], []
    for group in nhom_1:
        final_1.extend(group)
    final_1.extend(zero_group_1)
    final_1.extend(multiple_groups_1)

    for group in nhom_20:
        final_20.extend(group)
    final_20.extend(zero_group_20)
    final_20.extend(multiple_groups_20)

    final_1s = sorted(final_1, key=lambda x: x[0])
    final_20s = sorted(final_20, key=lambda x: x[0])

    final_answers_1 = []
    for i, (left_x, top_y) in enumerate(final_1s):
        for answer, group in nhomdau.items():
            if (left_x, top_y) in group:
                final_answers_1.append((i + 1, answer))
                break

    final_answers_20 = []
    for i, (left_x, top_y) in enumerate(final_20s):
        for answer, group in nhomcuoi.items():
            if (left_x, top_y) in group:
                final_answers_20.append((i + 21, answer))
                break

    final_answers = final_answers_1 + final_answers_20
    file_path = Path("answers.txt")
    with file_path.open(mode='w', encoding='utf-8') as file:
        file.write(str(final_answers))



    # Đọc nội dung từ original_answers.txt và answers.txt
    original_answers = read_answers('DOAN/original_answers.txt')
    user_answers = read_answers('answers.txt')

    # So sánh và in đáp án sai ra màn hình, tính điểm
    total_questions = len(original_answers)
    deduction_per_wrong_answer = 0.25

    score = 10
    wrong_answers = 0

    for (question, correct_answer), (_, user_answer) in zip(original_answers, user_answers):
        if correct_answer != user_answer:
            # print(f"Câu {question} sai. Đáp án đúng là {correct_answer}.")
            wrong_answers += 1

    # Tính điểm cuối cùng
    score -= wrong_answers * deduction_per_wrong_answer
    # print(f"Tổng số câu sai: {wrong_answers}")
    print(f"Kết quả: {score} điểm")
    return score
    # return wrong_answers

# Create a FastAPI and Detector instances
app = FastAPI()

@app.post("/origin/")
async def detect_on_img(file: UploadFile = File(...  )):
    """Endpoint to perform object detection on an uploaded image."""

    # # Create an instance of DetectionResults to store the detection results
    results = DetectionOrg()
    
    # # Read the uploaded image and decode it using OpenCV
    image = read_image(await file.read())

    cv2.imwrite(r"DOAN\upload_image.jpg",image)

    results.all_answers = final(image)
    return results

@app.post("/auto_grade/")
async def detect_on__st_img(file: UploadFile = File(...  )):

    rs_final = AutoGrade()
    
    # # Read the uploaded image and decode it using OpenCV
    img = read_image(await file.read())

    cv2.imwrite("img_st.jpg", img)

    xcrop1, xcrop2, ycrop1, ycrop2 = part_1(image_path = "img_st.jpg") 

    model = YoloDetect()
    detect = False
    img_crop = cv2.imread("img_crop.jpg")   
    frame = model.detect(frame=img_crop)
    detect = True
    
    rs_final.score = a(xcrop1, xcrop2, ycrop1, ycrop2, output_file_path = 'output.txt')
    # print(f"Tổng số câu sai: {rs_final.wrong_answers}")
    # print(f"Kết quả: {rs_final.score} điểm")

    return rs_final


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

