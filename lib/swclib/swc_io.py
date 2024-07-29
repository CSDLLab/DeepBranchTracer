import os
# from neuron_tracing.lib.pyneval.metric.utils import cli_utils
from lib.swclib.exceptions import InvalidSwcFileError
from lib.swclib.swc_tree import SwcTree
from anytree import NodeMixin, RenderTree, iterators
import numpy as np

def is_swc_file(file_path):
    return file_path[-4:] in (".swc", ".SWC")


def read_swc_tree(swc_file_path):
    if not os.path.isfile(swc_file_path) or not is_swc_file(swc_file_path):
        raise InvalidSwcFileError(swc_file_path)
    swc_tree = SwcTree()
    swc_tree.load(swc_file_path)
    return swc_tree

def read_swc_tree_matrix(swc_file_path):
    if not os.path.isfile(swc_file_path) or not is_swc_file(swc_file_path):
        raise InvalidSwcFileError(swc_file_path)
    swc_tree = SwcTree()
    swc_tree_matrix = swc_tree.load_matrix(swc_file_path)
    return swc_tree_matrix



# if path is a folder
# def read_swc_trees(swc_file_paths, tree_name_dict=None):
#     """
#     Read a swc tree or recursively read all the swc trees in a fold
#     Args:
#         swc_file_paths(string): path to read swc
#         tree_name_dict(dict): a map for swc tree and its file name
#             key(SwcTree): SwcTree object
#             value(string): name of the swc tree
#     Output:
#         swc_tree_list(list): a list shaped 1*n, containing all the swc tree in the path
#     """
#     # get all swc files
#     swc_files = []
#     if os.path.isfile(swc_file_paths):
#         if not is_swc_file(swc_file_paths):
#             print(swc_file_paths + "is not a swc file")
#             return
#         swc_files = [swc_file_paths]
#     else:
#         for root, _, files in os.walk(swc_file_paths):
#             for file in files:
#                 f = os.path.join(root, file)
#                 if is_swc_file(f):
#                     swc_files.append(f)
#     # load swc trees
#     swc_tree_list = []
#     for swc_file in swc_files:
#         swc_tree = SwcTree()
#         swc_tree.load(swc_file)
#         swc_tree_list.append(swc_tree)
#         if tree_name_dict is not None:
#             tree_name_dict[swc_tree] = os.path.basename(swc_file)
#     return swc_tree_list


# def adjust_swcfile(swc_str):
#     words = swc_str.split("\n")
#     return words
#
#
# def read_from_str(swc_str):
#     swc_tree = swc_tree.SwcTree()
#     swc_list = adjust_swcfile(swc_str)
#     swc_tree.load_list(swc_list)
#     return swc_tree


def swc_save(swc_tree, out_path, extra=None):
    out_path = os.path.normpath(out_path)
    swc_tree.sort_node_list(key="compress")
    swc_node_list = swc_tree.get_node_list()

    if not os.path.exists(os.path.dirname(out_path)):
        os.mkdir(os.path.dirname(out_path))

    with open(out_path, "w") as f:
        f.truncate()
        if extra is not None:
            f.write(extra)
        for node in swc_node_list:
            if node.is_virtual():
                continue
            try:
                f.write(
                    "{} {} {} {} {} {} {}\n".format(
                        node.get_id(),
                        node._type,
                        node.get_x(),
                        node.get_y(),
                        node.get_z(),
                        node.radius(),
                        node.parent.get_id(),
                    )
                )
            except:
                continue
    return True


def swc_save_preorder(in_path, out_path):
    if not os.path.exists(os.path.dirname(out_path)):
        os.mkdir(os.path.dirname(out_path))

    if not os.path.isfile(in_path) or not is_swc_file(in_path):
        raise InvalidSwcFileError(in_path)

    swc_tree = SwcTree()
    swc_tree.load(in_path)



    node_id_transform = np.zeros([swc_tree.size() ,2])
    node_id_temp = 1
    niter = iterators.PreOrderIter(swc_tree._root)
    for tn in niter:
        if tn.is_virtual():
            continue
        node_id_transform[node_id_temp - 1][0] = node_id_temp
        node_id_transform[node_id_temp - 1][1] = tn.get_id()
        # print(node_id_temp, tn.get_id())
        node_id_temp = node_id_temp + 1

    # print("=======================")
    niter = iterators.PreOrderIter(swc_tree._root)
    node_id_temp = 1
    with open(out_path, 'w') as fp:
        node_id_temp = 1
        for tn in niter:
            if tn.is_virtual():
                continue
            node_id_p_temp = tn.parent.get_id()
            # print(node_id_temp,tn.get_id())

            # print(tn.get_id(), tn._type, tn.get_x(), tn.get_y(), tn.get_z(), tn.radius(), node_id_p_temp)

            if node_id_p_temp == -1:
                node_id_p = -1
            else:
                ss = np.where(node_id_transform[:,1] == node_id_p_temp)
                node_id_p = node_id_transform[ss[0][0]][0]

            fp.write("{} {} {} {} {} {} {}\n".format(node_id_temp, tn._type, tn.get_x(), tn.get_y(), tn.get_z(),
                                                tn.radius(), node_id_p))
            node_id_temp = node_id_temp + 1
        fp.close()



def swc_save_metric(swc_data, save_dir):
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    with open(save_dir, 'w') as fp:
        for i in range(swc_data.shape[0]):
            fp.write('%d %d %g %g %g %g %d\n' % (
                swc_data[i][0], swc_data[i][1], swc_data[i][2], swc_data[i][3], swc_data[i][4], swc_data[i][5],
                swc_data[i][6]))
        fp.close()



        # if key == "PreOrderIter":
        #     niter = iterators.PreOrderIter(self._root)
        #     for tn in niter:
        #         if tn.get_id() == nid:
        #             return tn
        #
        #
        #         return None


if __name__ == "__main__":
    from pyneval.model import swc_node
    tree = SwcTree()
    tree.load("E:\\04_code\\00_neural_reconstruction\PyNeval\data\default.0.swc")
    swc_save(tree, "E:\\04_code\\00_neural_reconstruction\PyNeval\data\default.1.swc")