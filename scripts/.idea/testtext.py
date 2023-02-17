import cv2

# path being defined from where the system will read the image
path = r'test.png'
# command used for reading an image from the disk disk, cv2.imread function is used
image1 = cv2.imread(path)
# Window name being specified where the image will be displayed
window_name1 = 'image'
# font for the text being specified
font1 = cv2.FONT_HERSHEY_SIMPLEX
# org for the text being specified
org1 = (50, 50)
# font scale for the text being specified
fontScale1 = 1
# Blue color for the text being specified from BGR
color1 = (255, 255, 255)
# Line thickness for the text being specified at 2 px
thickness1 = 2
# Using the cv2.putText() method for inserting text in the image of the specified path
image_1 = cv2.putText(image1, 'CAT IN BOX', org1, font1, fontScale1, color1, thickness1, cv2.LINE_AA)
# Displaying the output image
cv2.imshow(window_name1, image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
