import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    os.chdir('filmrole3_frames')

    filenames = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
    filenames.sort()

    for filename in filenames:
        if '.jpg' in filename:
            img = plt.imread(filename)

            plt.imsave('../cropped/' + 'cropped_' + filename[:4] + filename[5:], img[89:1029,:,:])
