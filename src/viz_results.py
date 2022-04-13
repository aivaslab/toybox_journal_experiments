"""
Module for drawing graphs
"""
import matplotlib.pyplot as plt


def draw_instance_diversity_graph(instances_acc_list_in12, instances_acc_list_in12_wft, instances_acc_list_tb_1,
                                  instances_acc_list_tb_2, x_markers):
    """
    This method draws plt graph for instance diversity list
    """
    plt.plot(x_markers, instances_acc_list_in12, label="IN-12", marker=".")
    plt.xlabel("Number of instances per class")
    plt.ylabel("Test accuracy")
    plt.xticks(x_markers)
    plt.ylim([0, 100])
    plt.plot(x_markers, instances_acc_list_in12_wft, label="IN-12 (No FT)", marker=".")
    plt.plot(x_markers, instances_acc_list_tb_1, label="Toybox", marker=".")
    plt.plot(x_markers, instances_acc_list_tb_2, label="Toybox (FC-All)", marker=".")
    plt.legend(loc="upper right")
    plt.show()
    
    
def draw_view_diversity_graph(views_acc_list_6, views_acc_list_12, views_acc_list_24,
                              x_markers):
    """
    This method draws plt graph for instance diversity list
    """

    ax = plt.axes()
    plt.plot(x_markers, views_acc_list_6, label="6 instances", marker=".")
    plt.xlabel("Number of images per instance")
    plt.ylabel("Test accuracy on IN-12")
    
    plt.ylim([0, 60])

    plt.plot(x_markers, views_acc_list_12, label="12 instances", marker=".")
    plt.plot(x_markers, views_acc_list_24, label="24 instances", marker=".")

    plt.legend(loc="upper right")
    ax.set_xscale('log', base=2)
    ax.set_xticks(x_markers)
    ax.set_xticklabels(x_markers)
    plt.show()
    

if __name__ == "__main__":
    instances_acc_in12 = [34.417, 38.67, 39.58, 42.08, 43.17, 44.99]
    instances_acc_in12_wft = [8.5, 11.67, 16.25, 12.5, 16.25, 16.17]
    instances_acc_tb_1 = [28.49, 32.24, 42.41, 48.07, 58.99, 58.98]
    instances_acc_tb_2 = [41.41, 42.96, 47.21, 54.4, 58.41, 59.22]
    x_list = [1, 5, 10, 20, 25, 27]
    # draw_instance_diversity_graph(instances_acc_list_in12=instances_acc_in12, x_markers=x_list,
    #                               instances_acc_list_tb_1=instances_acc_tb_1,
    #                               instances_acc_list_tb_2=instances_acc_tb_2,
    #                               instances_acc_list_in12_wft=instances_acc_in12_wft)
    views_acc_6 = [33.25, 35.75, 37.58, 35.67, 38.67]
    views_acc_12 = [35.83, 42.08, 40.17, 39.25, 38.42]
    views_acc_24 = [41.83, 41.25, 44.17, 41.58, 42.58]
    x_list = [20, 40, 80, 160, 320]
    draw_view_diversity_graph(views_acc_list_6=views_acc_6, views_acc_list_12=views_acc_12,
                              views_acc_list_24=views_acc_24, x_markers=x_list)
    