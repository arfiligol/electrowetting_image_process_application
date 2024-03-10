import cv2


# Read the Image
img = cv2.imread("test_image.jpg")

# Show the Image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()