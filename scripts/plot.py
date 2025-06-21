import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, keypoints, outputs, img_size=(256, 256)):
    image = image.cpu().numpy()

    if image.shape[0] == 1:  # grayscale
        image = image.squeeze(0)
        plt.imshow(image, cmap='gray')
    else:  # RGB
        image = (image + 1) /2
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)

    # Unnormalize keypoints (if needed)
    keypoints = keypoints.cpu().numpy().reshape(-1, 2) * 224

    for x, y in keypoints:
        plt.scatter(x, y, c='r', s=10)


    outputs = outputs.cpu().numpy().reshape(-1, 2) * 224
    print(outputs)
    for x, y in outputs:
        plt.scatter(x, y, c='g', s=10)

    plt.axis('off')
    plt.show()
