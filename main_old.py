import cv2
from tkinter import *
from PIL import Image, ImageTk

# Define functions for Tkinter GUI
def update_canny(): # Used for changing the threshold of the canny algorithm
    # Get the current value of threshold
    low_threshold = low_slider.get()
    high_threshold = high_slider.get()
    cv2.GaussianBlur()

    # Apply the new value to the Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Find the contours and draw them
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(contour_img, ellipse, (255, 0, 0), 2)
    
    # Convert the processed image into Tkinter's format and resize
    contour_img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(contour_img_rgb)
    img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS) # Resize the image
    
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update the display
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

# Read the Image
img = cv2.imread("./test_image.jpg")

# Turn the image into gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create the main window of Tkinter
root = Tk()
root.title("Canny Edge Detection")

# Calculate the new dimensions to display the image
width, height = img.shape[1]*4, img.shape[0]*4

# Create canvas with the enlarged size
canvas = Canvas(root, width=width, height=height)
canvas.pack()

# Create the sliders for adjusting Canny's threshold
low_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Canny Low Threshold", command=lambda event: update_canny(), length=400)
low_slider.set(50) # Default value
low_slider.pack()

high_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Canny High Threshold", command=lambda event: update_canny(), length=400)
high_slider.set(150) # Default value
high_slider.pack()


# Initialize the image display
update_canny()

# Start loop
root.mainloop()
