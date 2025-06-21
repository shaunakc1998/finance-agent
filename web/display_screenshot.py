# display_screenshot.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_image(image_path):
    """Display an image using matplotlib"""
    print(f"Displaying image: {image_path}")
    
    # Load the image
    img = mpimg.imread(image_path)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.title('Finance Agent Chat UI')
    
    # Save as a new image for easier viewing
    output_path = 'screenshot_display.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved display image to: {output_path}")
    
    # Show the image
    plt.show()

if __name__ == "__main__":
    # Display the screenshot
    display_image("screenshot.png")
