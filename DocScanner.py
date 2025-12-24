import cv2
import numpy as np

def docsDetector(img):
    blur = cv2.GaussianBlur(img, (5, 5), 1.4)
    edges = cv2.Canny(blur,100,200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    doc_candidates = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            doc_candidates.append(approx)
    max_area = 0
    if len(doc_candidates) == 0:
        print("No document contour found")
        return
    else:
        document = doc_candidates[0]
        for i in doc_candidates:
            area = cv2.contourArea(i)
            if max_area < area:
                document = i
                max_area = area

    # VISUALISASI
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output, [document], -1, (0, 255, 0), 3)
    for point in document:
        x, y = point[0]
        cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
    cv2.imshow("Detected Document", output)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
    return document

def order_points(pts):
    pts = pts.reshape(4, 2)
    corners = np.zeros((4, 2), dtype=np.float64)

    maxidx = 0
    minidx = 0
    for i in range(1, len(pts)):
        s = pts[i][0] + pts[i][1]
        if(s < pts[minidx][0] + pts[minidx][1]):
            minidx = i
        if(s > pts[maxidx][0] + pts[maxidx][1]):
            maxidx = i
    corners[0] = pts[minidx]
    corners[2] = pts[maxidx]

    maxidx = 0
    minidx = 0
    for i in range(1, len(pts)):
        diff = pts[i][0] - pts[i][1]
        if(diff < pts[minidx][0] - pts[minidx][1]):
            minidx = i
        if(diff > pts[maxidx][0] - pts[maxidx][1]):
            maxidx = i
    corners[1] = pts[maxidx]
    corners[3] = pts[minidx]

    return corners

def width_height(corners):
    topwidth = np.linalg.norm(corners[1] - corners[0])
    bottomwidth = np.linalg.norm(corners[3] - corners[2])
    leftheight = np.linalg.norm(corners[0] - corners[3])
    rightheight = np.linalg.norm(corners[1] - corners[2])
    return int(max(topwidth, bottomwidth)), int(max(leftheight, rightheight))

def compute_homography(src, dst):
    A = np.zeros((8, 9), dtype=np.float64)
    if len(src) != len(dst) or len(src) != 4:
        print("Invalid number of points")
        return
    for i in range(len(src)):
        x = src[i][0]
        y = src[i][1]
        u = dst[i][0]
        v = dst[i][1]
        ax = np.array([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        ay = np.array([0, 0, 0, -x, -y, -1, v*x, v*y, v])
        A[2*i] = ax
        A[2*i+1] = ay
    
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1,:].flatten()
    H = h.reshape((3, 3))
    H = H / H[2, 2]
    return H

def warp_image(image, H, width, height):
    Hinverse = np.linalg.inv(H)
    output = np.zeros((height, width, 3), dtype = image.dtype)
    for v in range(height):
        for u in range(width):
            out_coord = np.array([u, v, 1.0])
            x, y, w = np.dot(Hinverse, out_coord)
            if(w != 0):
                x /= w
                y /= w
                if (0 <= int(y) <= image.shape[0]) and (0 <= int(x) <= image.shape[1]):
                    output[v, u] = image[int(y), int(x)]
    return output

image = input("Masukkan nama file: ")
img = cv2.imread(image)
img_grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
corners = docsDetector(img_grayscale)
if corners is None or len(corners) != 4:
    raise ValueError("Document corners not detected")
corners = order_points(corners)
width, height = width_height(corners)
out_corners = np.array([ [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1] ], dtype=np.float64)
H = compute_homography(corners, out_corners)
output = warp_image(img, H, width, height)
cv2.imshow("Warped Document", output)
key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
