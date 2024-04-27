import cv2
import imutils
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to process the image and extract the number plate
def process_image():
    try:
        # Open an image file dialog to select the image
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        # Read the selected image
        image = cv2.imread(file_path)

        # Resize the image
        image = imutils.resize(image, width=500)
        #convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #binary thresholding
        ret1,bina = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
        #smoothing using bilateral filter
        smoothed = cv2.bilateralFilter(bina, 11, 17, 17)
        #bluring using gaussian blur
        blur =cv2.GaussianBlur(smoothed,(5,5),0)


        # Edge detection
        #? if the pixel is less than threshold_1 it is discarded and if the pixel is higher than threshold_2 it is considerd an edge.
        edged = cv2.Canny(blur, 170, 200)

        # Contour detection
        #?cnts like a curve joining all the continous points
        #?RETR_LIST - it retrives all the contours but desn't create any parent-child relationship
        #?CHAIN_APPROX_SIMPLE -It removes all the redundant points and compress the contour by saving memory
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        #? Sort the contour list according to the contourArea.Then slice the first 30 of it,
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

        # Initialize variables for cropped image and text
        cropped_image = None
        text = ""

        #?loop through the sorted cnts array.
        for i in cnts:
            #?calcultes the perimater of the contour.
            perimeter = cv2.arcLength(i, True)
            #?It approximate the contour as a simple polygon and the accuracy is given by 0.002*perimeter.
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            #?If the simplified polygon has four points it is assumed to be the liscense palte.
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(i)
                cropped_image = image[y:y+h, x:x+w]

                # Extract text from the number plate
                #?This configuration detects the capital letters and the numbers.
                text = pytesseract.image_to_string(Image.fromarray(cropped_image), config='--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                
                break

        if cropped_image is not None:
            # Display the cropped plate image within the Tkinter app
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(cropped_image))
            image_label.config(image=photo)
            image_label.image = photo

            # Show the extracted text in a message box
            messagebox.showinfo("Number Plate", "Number is: " + text)
    
    except Exception as e:
        messagebox.showerror("Error", "An error occurred: " + str(e))

# Create a GUI window
window = tk.Tk()
window.title("License Plate Detection")
window.geometry("800x600")
window.configure(bg="light blue")

# Heading label
heading_label = tk.Label(window, text="License Plate Detection System", font=("Helvetica", 16, "bold"), bg="light blue")
heading_label.pack(pady=20)

# Create a button to trigger image processing
process_button = tk.Button(window, text="Process Image", command=process_image, bg="gray", font=("Helvetica", 12), cursor="hand2")
process_button.pack(padx=20, pady=10)

# Image label to display the cropped license plate
image_label = tk.Label(window, bg="light blue")
image_label.pack(padx=20, pady=10)

# Create a button to exit the application
exit_button = tk.Button(window, text="Exit", command=window.quit, cursor="hand2", bg="gray", font=("Helvetica", 12))
exit_button.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()

