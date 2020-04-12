import matplotlib.pyplot as plt



fontsize=44
ticksize=36



def distributed_hsq():
    percents = [0, 77, 90, 0]
    x = [0, 0.65, 2.35, 3]
    plt.bar(x, percents, width=1.15,
            tick_label=["", "HSQ-SGD", "SGD", ""],
            fc='dimgray')


    plt.xlabel('Optimizer', fontsize=fontsize)
    plt.ylabel('Per Epoch Time (Minutes)', fontsize=fontsize)

    plt.text(x = x[1] - 0.12, y = percents[1] + 0.2, s = str(percents[1]), size = fontsize * 0.65, color='blue')
    plt.text(x = x[2] - 0.12, y = percents[2] + 0.2, s = str(percents[2]), size = fontsize * 0.65, color='blue')



    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.show()

if __name__ == '__main__':
    distributed_hsq()
