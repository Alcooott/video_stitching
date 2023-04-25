import cv2
import numpy as np
import time

def color_adjusted(image1,image2):
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

    #色调调为一致
    mean_hue1 = cv2.mean(hsv_image1[:, :, 0])[0]
    mean_hue2 = cv2.mean(hsv_image2[:, :, 0])[0]
    scale_factor_saturation = mean_hue1 / mean_hue2
    image2_saturation_channel = hsv_image2[:, :, 0]
    scaled_image2_saturation_channel = cv2.multiply(image2_saturation_channel, scale_factor_saturation)
    hsv_image2[:, :, 0] = scaled_image2_saturation_channel


    #饱和度调为一致
    mean_saturation1 = cv2.mean(hsv_image1[:, :, 1])[0]
    mean_saturation2 = cv2.mean(hsv_image2[:, :, 1])[0]
    scale_factor_saturation = mean_saturation1 / mean_saturation2
    image2_saturation_channel = hsv_image2[:, :, 1]
    scaled_image2_saturation_channel = cv2.multiply(image2_saturation_channel, scale_factor_saturation)
    hsv_image2[:, :, 1] = scaled_image2_saturation_channel

    #亮度调为一致
    mean_value1 = cv2.mean(hsv_image1[:, :, 2])[0]
    mean_value2 = cv2.mean(hsv_image2[:, :, 2])[0]
    scale_factor_value = mean_value1 / mean_value2
    image2_value_channel = hsv_image2[:, :, 2]
    scaled_image2_value_channel = cv2.multiply(image2_value_channel, scale_factor_value)
    hsv_image2[:, :, 2] = scaled_image2_value_channel

    image2 = cv2.cvtColor(hsv_image2, cv2.COLOR_HSV2BGR)

    return image1,image2

def sift(image):

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)

    return  kp, des

def get_good_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def video_sift(img_right, img_left):
    kp1, des1 = sift(img_right)
    kp2, des2 = sift(img_left)
    goodMatch = get_good_match(des1, des2)
    # 当筛选项的匹配对大于4对时：计算视角变换矩阵
    if len(goodMatch) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)
        result = cv2.warpPerspective(img_right, H, (img_right.shape[1] + img_left.shape[1], img_right.shape[0]))
        result[0:img_left.shape[0], 0:img_left.shape[1]] = img_left
        return result



video_1 = cv2.VideoCapture("./dataset/homework/left.mp4")
video_2 = cv2.VideoCapture("./dataset/homework/right.mp4")

width = int(video_1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
frames_num=video_1.get(cv2.CAP_PROP_FRAME_COUNT)
videoWriter = cv2.VideoWriter('C:\\Users\\stark\\Desktop\\大三\\Computer_vision\\exp\\ouput\\output_2.mp4', fourcc, fps, (width * 2, height))

num=0
while True:
    # 读取视频
    (ret_1, frame1) = video_1.read()
    (ret_2, frame2) = video_2.read()
    if ret_1 and ret_2:


        time_start=time.perf_counter()
        frame1,frame2 = color_adjusted(frame1, frame2)
        result = video_sift(frame2, frame1)

        time_end=time.perf_counter()
        time_sum=time_end-time_start
        print("The time taken to process NO.%s frame is %ss"%(str(num),str(time_sum)))

        cv2.imwrite(r"./ouput/frames/%s.jpg"%str(num),result)
        videoWriter.write(result)
        num += 1
        cv2.namedWindow("result",0)
        cv2.imshow("result", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

videoWriter.release()
video_1.release()
video_2.release()
cv2.destroyAllWindows()