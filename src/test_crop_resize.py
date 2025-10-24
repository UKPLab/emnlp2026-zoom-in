from PIL import Image

image_path = ""

img = Image.open('test.jpg')
img = img.crop((0, 0, 100, 100))
img = img.resize((200, 200))
img.save('test_crop_resize.jpg')