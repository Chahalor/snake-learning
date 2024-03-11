import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores:list, mean_scores:list) -> None :
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training :)")
    plt.xlabel("number of game")
    plt.ylabel("score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])
             )
