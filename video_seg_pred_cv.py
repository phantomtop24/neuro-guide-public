from ultralytics import YOLO
import numpy as np
import time
import cv2
import os

# Функция продолжительности
def video_time(video_list):
    
    duration = 0
    frames = 0
    for video_name in video_list:
        video = cv2.VideoCapture(os.path.join('sidewalk_detection_learning', 'check_video', video_name))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration += int(frame_count / fps)
        frames += int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
    return duration, frames

# Метод отрисовки
def cv2_otrisovka_func(coords, frame, class_id):
    
    if class_id == 'car' or class_id == 'bicycle' or class_id == 'motorcycle' or class_id == 'bus' or class_id == 'truck':
        color = (0, 0, 145)
    elif class_id == 'person':
        color = (0, 130, 255)
    elif class_id == 'sign':
        color = (130, 0, 75)
    elif class_id == 'wait':
        color = (0, 0, 255)
    elif class_id == 'cross':
        color = (0, 200, 0)
    else:
        color = (200, 200, 200)
        
    x1, y1, x2, y2 = coords
        
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.putText(frame, class_id, (int(x1), int((y1 - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# Модели
model_sidewalk = YOLO('best_sidewalk.pt')
model_zebra = YOLO('best_zebra.pt')
model_sign = YOLO('best_sign.pt')
model_standart = YOLO('yolov10n.pt')

# Список видео
video_list = ['video.mp4']

duration, frames = video_time(video_list)
print(f"Примерная продолжительность обработки видео: {round(duration * 2.5)}c")
time.sleep(3)
start_time = time.time()

# Цикл, по очереди перебирающий каждое видео из списка видео
sign_recharge = 0
video_num = 1
wait_time = 60
cross_time = 60
sign_time = 30
frames_for_percent = 0
for video_name in video_list:
    window = cv2.VideoCapture(os.path.join('sidewalk_detection_learning', 'check_video', video_name))
    ret, frame = window.read()
    h, w = frame.shape[:2]
    frame_num = 0
    out = cv2.VideoWriter('results\\{}_out(v1.11).mp4'.format(video_name.replace('.mp4', '')), cv2.VideoWriter_fourcc(*'MP4V'), int(window.get(cv2.CAP_PROP_FPS)), (w, h))

    # Цикл для обработки каждого видео
    while ret:
        # Условие обработки каждого 4-ого кадра (для оптимизации программы)
        if frame_num == 1 or frame_num % 4 == 0:
            results_sidewalk = model_sidewalk.predict(frame, show_boxes=False, save=False, agnostic_nms=True, iou=0.5)
            results_zebra = model_zebra.predict(frame, show_boxes=False, save=False, agnostic_nms=True, iou=0.5)
            results_sign = model_sign.predict(frame, save=False, agnostic_nms=True, iou=0.5)
            results_standart = model_standart.predict(frame, save=False, classes=[0, 1, 2, 3, 5, 7], agnostic_nms=True, iou=0.5)
            Classes = []
            
            # Обработка и отрисовка тротуаров на frame
            for obj_num in range(len(results_sidewalk[0].boxes)):
                mask_sidewalk = results_sidewalk[0].masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                class_id = results_sidewalk[0].names[results_sidewalk[0].boxes.cls[obj_num].item()]
                cv2.drawContours(frame, [mask_sidewalk], 0, (150, 0, 0), 5)
            
            # Обработка и отрисовка зебры на frame
            for obj_num in range(len(results_zebra[0].boxes)):
                mask_zebra = results_zebra[0].masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                class_id = results_zebra[0].names[results_zebra[0].boxes.cls[obj_num].item()]
                cv2.drawContours(frame, [mask_zebra], 0, (150, 0, 0), 5)
                Classes.append(class_id)
            
            # Обработка и отрисовка знака, 'красного' и 'зеленого света' на frame
            for obj_num in range(len(results_sign[0].boxes)):
                box = results_sign[0].boxes
                class_id = results_sign[0].names[box.cls[obj_num].item()]
                coords = sum(box[obj_num].xyxy.tolist(), [])
                cv2_otrisovka_func(coords, frame, class_id)
                Classes.append(class_id)
            
            # Обработка и отрисовка человека, машины и велосипеда на frame
            for obj_num in range(len(results_standart[0].boxes)):
                box = results_standart[0].boxes
                class_id = results_standart[0].names[box.cls[obj_num].item()]
                coords = sum(box[obj_num].xyxy.tolist(), [])
                cv2_otrisovka_func(coords, frame, class_id)
        
        else:
            # Отрисовка тротуаров на frame
            for obj_num in range(len(results_sidewalk[0].boxes)):
                cv2.drawContours(frame, [mask_sidewalk], 0, (150, 0, 0), 5)
            
            # Отрисовка зебры на frame
            for obj_num in range(len(results_zebra[0].boxes)):
                cv2.drawContours(frame, [mask_zebra], 0, (150, 0, 0), 5)
            
            # Отрисовка знака, 'красного' и 'зеленого света' на frame
            for obj_num in range(len(results_sign[0].boxes)):
                box = results_sign[0].boxes
                class_id = results_sign[0].names[box.cls[obj_num].item()]
                coords = sum(box[obj_num].xyxy.tolist(), [])
                cv2_otrisovka_func(coords, frame, class_id)
            
            # Отрисовка человека, машины и велосипеда на frame
            for obj_num in range(len(results_standart[0].boxes)):
                box = results_standart[0].boxes
                class_id = results_standart[0].names[box.cls[obj_num].item()]
                coords = sum(box[obj_num].xyxy.tolist(), [])
                cv2_otrisovka_func(coords, frame, class_id)
                
        # Проверка на возможность перехода дороги в данном месте
        if 'wait' in Classes and not('cross' in Classes) and 'zebra' in Classes:
            wait_time = 0
        elif 'cross' in Classes and not('wait' in Classes) and 'zebra' in Classes:
            cross_time = 0
        elif 'sign' in Classes and not('wait' in Classes) and not('cross' in Classes) and not('zebra' in Classes) and sign_recharge > 240:
            sign_time = 0
            sign_recharge = 0

        # Отрисовка предупреждений и советов
        if wait_time < 60:
            cv2.putText(frame, 'Wait', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            wait_time += 1
        elif cross_time < 60:
            cv2.putText(frame, 'Cross', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 5, cv2.LINE_AA)
            cross_time += 1
        elif sign_time < 30:
            cv2.putText(frame, 'The pedestrian crossing nearby', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (130, 0, 75), 5, cv2.LINE_AA)
            sign_time += 1

        # Запись кадра
        out.write(frame)
        
        # Обновление кадра на следующий из видео
        ret, frame = window.read()  
        frames_for_percent += 1
        sign_recharge += 1
        frame_num += 1
        
        # Отображение номера видео и общего процента выполнения обработки
        print(f"                                                                                                     Видео: {video_num}/{len(video_list)}  {round(frames_for_percent/frames*100)}%")
    video_num += 1

# Фиксация и вывод времени обработки всех видео
end_time = time.time()
print(f"Время, понадобившееся для обработки видео: {round(end_time - start_time)}c")


window.release()
out.release()
cv2.destroyAllWindows()