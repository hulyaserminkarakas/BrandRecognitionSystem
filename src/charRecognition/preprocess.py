from PIL import Image


def main():
    for i in range(1, 6284):
        img = Image.open(f"train_greyscale/{i}", 'r')
        img.save(f"train_greyscale/{i}.png", format='png')


if __name__ == '__main__':
    main()
