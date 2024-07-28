import cv2
import numpy as np
from sklearn.cluster import KMeans
import imutils

def color_of_image(filepath):
    clusters = 3 
    img = cv2.imread(filepath)
    org_img = img.copy()
    img = imutils.resize(img, height=200)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]

    p_and_c = list(zip(percentages, dominant_colors))
    p_and_c = sorted(p_and_c, reverse=True)

    # Kiểm tra xem có màu chủ đạo nằm trong khoảng màu đỏ không
    red_lower = np.array([0, 0, 100], dtype='uint')
    red_upper = np.array([100, 100, 255], dtype='uint')
    is_red_dominant = any((red_lower <= color).all() and (color <= red_upper).all() for _, color in p_and_c)

    yellow_lower = np.array([0, 150, 160], dtype='uint')
    yellow_upper = np.array([100, 255, 255], dtype='uint')
    is_yellow_dominant = any((yellow_lower <= color).all() and (color <= yellow_upper).all() for _, color in p_and_c)
    
    green_lower = np.array([0, 80, 0], dtype='uint')
    green_upper = np.array([50, 255, 50], dtype='uint')
    is_green_dominant = any((green_lower <= color).all() and (color <= green_upper).all() for _, color in p_and_c)
    print(is_yellow_dominant, is_red_dominant, is_green_dominant)
    
    if not is_green_dominant:
        print(1) 
    else:
        print(0) 

    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows//2-250, cols//2-90), (rows//2+100, cols//2+110), (255, 255, 255), -1)

    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors', (rows//2-230, cols//2-40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    start = rows//2-220
    for i in range(3):
        end = start + 70
        final[cols//2:cols//2+70, start:end] = p_and_c[i][1]
        print(p_and_c[i][1])
        cv2.putText(final, str(i+1), (start+25, cols//2+45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        start = end + 20

    cv2.imshow('Dominant Colors', final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filepath = "a.jpg"
color_of_image(filepath)
