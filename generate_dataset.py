import os

import cv2
import numpy as np

def test():
    sylas = cv2.imread("champion_icons/Sylas.jpg")
    cv2.imshow("sylas_gray", cv2.resize(sylas, (300, 300)))
    sylas = cv2.resize(cv2.resize(sylas, (25, 25)), (300, 300))
    cv2.imshow("sylas_small", sylas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def progress_bar(l, convert=False):
    if convert:
        l = list(l)
    for d, i in enumerate(l):
        if type(l) == list:
            print(f"\r{d/len(l)*100:.2f}% done!", end="")
        else:
            print(f"\r{d} done!", end="")
        yield i
    print("\rDone!        ")


def get_bounds(image, evaluator, accurate=False):
    bounds = []
    for row in image:
        this = []
        for count, pixel in enumerate(row):
            if accurate:
                if not evaluator(pixel):
                    this.append(count)
                continue
            if len(this) == 0:
                if not evaluator(pixel):
                    this.append(count)
            else:
                if evaluator(pixel):
                    this.append(count)
                    if not accurate:
                        break
            if len(this) == 0 and count == len(row) - 1:
                this.append(-1)
        bounds.append(this)
    return bounds


def alpha_channel_compare(pixel):
    return pixel[3] < 240


def is_white(pixel):
    if pixel.shape:
        white = np.full(pixel.shape[-1], 255, dtype=np.uint8)
        return np.all(pixel == white)
    return pixel == 255


def is_almost_white(pixel):
    if pixel.shape:
        white = np.full(pixel.shape[-1], 200, dtype=np.uint8)
        return np.all(pixel >= white)
    return pixel > 200


class ImageBounds:
    image = None
    bounds = None
    name = "not specified"
    accurate = False

    def __repl__(self):
        return self.name


class Champion(ImageBounds):
    def __init__(self, champ_id, name):
        self.id = champ_id
        self.name = name
        self.image = cv2.resize(cv2.imread(f'gendataset/champion_icons/{name}.jpg', cv2.IMREAD_UNCHANGED), (25, 25))
        self.bounds = get_bounds(self.image, alpha_channel_compare)
        self.width, self.height = self.image.shape[1::-1]
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)


class Ping(ImageBounds):

    def __init__(self, name, **kwargs):
        self.image = cv2.imread(f"gendataset/pings/{name}", cv2.IMREAD_UNCHANGED)
        self.bounds = get_bounds(self.image, is_almost_white, **kwargs)
        if "accurate" in kwargs:
            self.accurate = kwargs["accurate"]
        self.width, self.height = self.image.shape[1::-1]
        # if name.startswith("q"):
            # print(name, self.bounds, self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)


class DatasetGenerator:
    """
    Class that is used to generate the dataset
    """

    def __init__(self):
        self.champions = []
        self.pings = []
        self.image_location = 'gendataset/champion_icons'
        self.training_folder = 'datasets/lol/'
        self.initialize()
        self.minimap = cv2.imread('gendataset/final_minimap.png', cv2.IMREAD_UNCHANGED)
        self.load_pings()

    def initialize(self):
        if not os.path.exists(self.training_folder):
            os.makedirs(self.training_folder)
        # make images and labels if they don't exist
        for folder in ["images", "labels"]:
            if not os.path.exists(self.training_folder + folder):
                os.makedirs(self.training_folder + folder)
            # make train test val folders if they don't exist
            for subfolder in ["train", "test", "val"]:
                if not os.path.exists(self.training_folder + folder + "/" + subfolder):
                    os.makedirs(self.training_folder + folder + "/" + subfolder)
        
        
        for count, i in enumerate(os.listdir(self.image_location)):
            name = ".".join(i.split(".")[:-1])
            self.champions.append(Champion(count, name))
        # save champs to a yaml fiel
        yml = f"""
path: ./datasets/lol
train: images/train
val: images/val
test: images/test

nc: {len(self.champions)}
names: [
    {", ".join(['"{}"'.format(champ.name) for champ in self.champions])}
]"""
        with open("lol.yaml", "w") as file:
            file.write(yml)


    def load_pings(self):
        pings = {"assist.jpg": {}, "back.jpg": {}, 'blue.jpg': {}, 'danger.jpg': {}, "omw.jpg": {}, 'question.jpg': {}, "vision.jpg" : {"accurate": True}}
        for name, kwargs in pings.items():
            self.pings.append(Ping(name, **kwargs))

    def generate_dataset(self):
        """
        This function generates a dataset and then saves it with numpy
        :returns: None
        """
        wh = self.minimap.shape[0]
        champs_used = {champ:0 for champ in self.champions}
        for i in progress_bar(range(100000), True):
            # create 100000 images
            minimap = self.minimap.copy()
            # 20% chance of minimap being completely random noise
            random_num = np.random.randint(0, 100)
            if random_num < 20:
                minimap = np.random.randint(0, 255, size=minimap.shape, dtype=np.uint8)
            # 60% chance of having some noise
            if random_num < 80:
                noise = np.random.randint(0, 100, size=minimap.shape, dtype=np.uint8)
                # 50% chance of adding noise to the minimap
                if np.random.randint(0, 100) < 50:
                    minimap = cv2.add(minimap, noise)
                else:
                    minimap = cv2.subtract(minimap, noise)

            exclude = []
            box = []
            for ii in range(10):
                champion = self.get_random_champion_equally(champs_used, exclude)
                exclude.append(champion)
                pos = np.random.randint(0, wh-champion.width, 2)
                if ii < 5:
                    self.add_transparent(minimap, champion, pos)
                else:
                    self.add_transparent(minimap, champion, pos, blue=False)
                if np.random.random()<=0.1:
                    ping = np.random.choice(self.pings)
                    ping_pos = self.get_random_ping_location(minimap, champion, pos, ping)
                    self.add_transparent(minimap, ping, ping_pos)
                    if np.random.random()<=0.3:
                        ping = np.random.choice(self.pings)
                        ping_pos = self.get_random_ping_location(minimap, champion, pos, ping)
                        self.add_transparent(minimap, ping, ping_pos)
                box.append([champion.id, (pos[0]+25/2)/minimap.shape[1], (pos[1]+25/2)/minimap.shape[0], 25/minimap.shape[1], 25/minimap.shape[0]])

            folder_name = "test/"
            if i < 95000:
                folder_name = "train/"
            elif i < 97500:
                folder_name = "val/"

            # save image
            cv2.imwrite(f"{self.training_folder}images/{folder_name}{i}.jpg", minimap) 
            # save labels
            with open(f"{self.training_folder}labels/{folder_name}{i}.txt", 'w') as file:
                for bbox in box:
                    file.write(" ".join([str(iii) for iii in bbox]))
                    file.write("\n")
        

    @staticmethod
    def get_random_champion_equally(champs_used, exclude = None):
        champs = []
        min_used = 999999999999999
        for champ, used in champs_used.items():
            if exclude:
                if champ in exclude:
                    continue
            if used < min_used:
                champs = [champ]
                min_used = used
            elif used == min_used:
                champs.append(champ)
        champ = np.random.choice(champs)
        champs_used[champ] += 1
        return champ






    @staticmethod
    def get_random_ping_location(map, champion, pos, ping):
        """
        Return a random location for a ping

        :param map: image of the minimap
        :param champion: the champion
        :param pos: top-left position of the champion
        :param ping: the ping to draw on the minimap
        :returns: top-left position of the ping
        """
        mapW, mapH = map.shape[1::-1]
        x = pos[0] + np.random.randint(-ping.width + 5, champion.width - 5)
        if x + ping.width >= mapW:
            x = mapW - 1 - ping.width
        elif x <= 0:
            x = 0
        y = pos[1] + np.random.randint(-ping.height + 5, champion.height - 5)
        if y + ping.height >= mapH:
            y = mapH - 1 - ping.height
        elif y <= 0:
            y = 0
        return x, y

    def test(self):
        temp_map = self.minimap.copy()
        champion = self.champions[0]
        self.add_transparent(temp_map, champion, (80, 80))
        temp_map = cv2.cvtColor(temp_map, cv2.COLOR_RGBA2GRAY)
        #temp_map[40:65, 40:65] = champion.image
        p = Ping()
        self.add_transparent(temp_map, p, (80, 80))
        cv2.imshow("ping", p.image)
        #self.ping(temp_map, (95, 95), False)
        cv2.imshow("test", temp_map)

    @staticmethod
    def ping(img, pos, blue=True, gray=True):
        if gray:
            if blue:
                cv2.circle(img, pos, 5, 127, -1)
            else:
                cv2.circle(img, pos, 5, 53, -1)

        else:
            if blue:
                cv2.circle(img, pos, 5, (157, 160, 68), -1)
            else:
                cv2.circle(img, pos, 5, (29, 34, 136), -1)

    @staticmethod
    def add_transparent(image: np.array, template: ImageBounds, top_left: tuple, blue=True):
        """
        Draws a transparent image on another image

        :param image: the image to draw on
        :param template: the template that is drawn on the image
        :param top_left: top-left position of the template when drawn
        :param blue: if template is a champion, true ---> blue circle, false ---> red circle around the champion
        :returns:
        """
        x, y = top_left
        width = template.image.shape[1]
        if template.accurate:
            for count, bounds in enumerate(template.bounds):
                for i in range(width):
                    if i in bounds:
                        image[y + count, x + i] = template.image[count, i]

        else:
            for count, bounds in enumerate(template.bounds):
                if len(bounds) == 2:
                    left, right = bounds
                    image[y + count, x + left:x+right] = template.image[count, left:right]
                elif len(bounds) == 1:
                    left = bounds[0]
                    if left != -1:
                        image[y + count, x + left:x + width] = template.image[count, left:]
                else:
                    image[y + count, x:x+width] = template.image[count]
        if type(template) == Champion:
            if blue:
                cv2.circle(image, (int(x+width/2), int(y+width/2)), int(width/2), (0xE0, 0x99, 0x00), 1)
            else:
                cv2.circle(image, (int(x+width/2), int(y+width/2)), int(width/2), (0x3D, 0x3D, 0xE8), 1)
        return image




if __name__=="__main__":
    gen = DatasetGenerator()
    gen.generate_dataset()
    cv2.destroyAllWindows()
