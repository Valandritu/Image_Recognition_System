# import cv2

# # Load the image
# img = cv2.imread('input.jpg')

# # Set new dimensions (e.g., 2x bigger)
# scale_percent = 200  # Increase size by 200%
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)

# # Resize with interpolation method
# resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

# # Save the output
# cv2.imwrite('upscaled_image.jpg', resized_img)



import cv2

# Load the image
img = cv2.imread('input.jpg')

# Set new dimensions (e.g., 2x bigger)
scale_percent = 200  # Increase size by 200%
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize with interpolation method
resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

# Save the output
cv2.imwrite('upscaled_image.jpg', resized_img)