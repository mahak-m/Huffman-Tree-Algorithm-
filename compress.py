from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for i in text:
        if i in freq_dict:
            freq_dict[i] += 1
        else:
            freq_dict[i] = 1
    return freq_dict


def get_freq(item: tuple[int, int]) -> int:
    """
    this helper function returns the second value of the tuple
    representing the (frequency) of a character
    """
    return item[1]


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # sorting freq_dict by increasing frequency/values
    freq_dict = dict(sorted(freq_dict.items(), key=get_freq))

    # creating nested list <all_nodes>
    all_nodes = []
    for key, value in freq_dict.items():
        lst = [value, HuffmanTree(key, None, None)]
        all_nodes.append(lst)

    # case 1: if freq_dict is empty
    if len(all_nodes) == 0:
        return None

    # case 2: if freq_dict has only 1 element, return HuffmanTree obj
    if len(all_nodes) == 1:
        return HuffmanTree(None, all_nodes[0][1], all_nodes[0][1])

    # case 3: otherwise, construct the huffman tree
    else:
        output_tree = build_tree(leaves=all_nodes)
        return output_tree[0][1]


def get_first(t: list) -> int:
    """
    this helper function returns the first value of a list
    """
    return t[0]


def build_tree(leaves: list) -> list:
    """
    Combine smallest and second-smallest frequencies to create a
    new parent, add it to nested list <leaves> as a list
    Precondition: len(all_nodes) > 1
    """
    while len(leaves) > 1:
        # sort the leaves
        leaves.sort(key=get_first)
        # create new parent value with the sum of the 2 smallest values
        smallest = leaves.pop(0)
        second_smallest = leaves.pop(0)
        parent_val = int(smallest[0] + second_smallest[0])
        # add parent value and new Huffman tree into <leaves>
        curr = [parent_val, HuffmanTree(None, smallest[1], second_smallest[1])]
        leaves.append(curr)
    return leaves


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree is None:  # if None, return empty dict.
        return {}
    else:  # otherwise, use recursion to fill the dictionary
        return dict_helper(tree, {}, '')


def dict_helper(tree: HuffmanTree, dict_: dict, value: str) -> dict:
    """
    Helper method for <get_codes> to fill dictionary
    """
    # (Unsure where to initalize dict and update <value> w/out
    # adding new parameters, so helper function is created: fix)
    # base case
    if tree.left is None and tree.right is None:
        dict_[int(tree.symbol)] = str(value)  # if leaf, create dict
        # print('dict 2', dict)
        return dict_
    # recursive case
    if tree.left is not None:  # add 0 for lhs instead of ''
        dict_helper(tree.left, dict_, value + '0')
    if tree.right is not None:  # add 1 for rhs instead of ''
        dict_helper(tree.right, dict_, value + '1')
    # print('dict 3', dict)
    return dict_


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    counter = [0]
    # (counter is a list b/c ints are immutable)
    number_helper(tree, counter)


def number_helper(tree: HuffmanTree, counter: list) -> None:
    """
    Helper function for <number_nodes> that handles the base and recursive case.
    [A helper function is needed for <number_nodes> because a parameter cannot
    be added to the original method (counter must be a parameter to ensure
    that it doesn't reset with every recursive iteration)]
    """
    # base cases:
    # if it's None or just the root, it's not an internal node (pass)
    if tree is None:
        pass
    elif tree.left is None and tree.right is None:
        pass
    # recursive case:
    else:
        number_helper(tree.left, counter)
        number_helper(tree.right, counter)
        tree.number = counter[0]  # set number as <counter>
        counter[0] += 1  # append doesn't work(?) fix


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # to calculate the av., first find the node which contains the key
    # from <freq_dict>. then, we must find the length of that node
    if freq_dict == {}:
        return 0.0
    else:
        numerator = 0.0
        denomenator = 0.0
        for keys, values in freq_dict.items():
            symbol_ = (get_symbol(tree, freq_dict, keys))  # gets node
            length = length_helper(tree, symbol_)  # gets length of the node
            # print('get_symbol', get_symbol(tree, freq_dict, keys))
            numerator += values * length  # = frequencies * length
            denomenator += values  # = sum of the frequencies
        num = float(numerator / denomenator)
        return num


def get_symbol(tree: HuffmanTree, freq_dict: dict, symbol: int) -> HuffmanTree:
    """
    Helper method that gives the corresponding node for a key in <freq_dict>.
    It returns None if the symbol is not in <tree>
    """
    # base cases:
    if tree.symbol == symbol:
        return tree  # if the root is the symbol, return it
    elif tree.left is None and tree.right is None:
        return None  # return None if LHS & RHS is None
    # recursive case:
    # recurse until the symbol == root for some node
    else:
        left_node = get_symbol(tree.left, freq_dict, symbol)
        if left_node is not None:
            return left_node
        right_node = get_symbol(tree.right, freq_dict, symbol)
        if right_node is not None:
            return right_node
        return None  # means <symbol> is not in <tree>


def length_helper(tree: HuffmanTree, node: HuffmanTree, height: int = 0) -> int:
    """
    Helper function that calculates the length of a node in a Huffman
    Tree given the node
    """
    # base cases:
    if tree is None:  # the length is 0 if tree is None
        return 0
    elif tree.symbol == node.symbol:
        return height  # the length is <height>, initialized as 0
    # recursive case:
    else:
        # if tree.left exists, add 1 to height and store total length
        # in the variable <lhs>:
        if tree.left:
            lhs = length_helper(tree.left, node, height + 1)
        else:
            lhs = 0
        # do the same with tree.right and <rhs>:
        if tree.right:
            rhs = length_helper(tree.right, node, height + 1)
        else:
            rhs = 0
        # return whichever side has the greater height. ie. max(2,3) = 3
        return max(lhs, rhs)


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # first convert each byte from <codes> into -> bit -> list of bits
    bit_sequence = []
    for i in text:
        bit_sequence += codes[i]  # get the value from <codes> dict

    result = []
    # for every 8 bits, concatenate into string <byte>
    for i in range(0, len(bit_sequence), 8):
        byte_bits = bit_sequence[i:i + 8]
        # loop over every bit in the byte:
        byte = ""
        for bit in byte_bits:
            byte += str(bit)
        # use <bits_to_byte> to convert bits into bytes:
        byte_int = bits_to_byte(byte)
        result.append(byte_int)
    return bytes(result)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.
    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    all_bytes = []  # stores bytes
    create_list_helper(tree, all_bytes)
    return bytes(all_bytes)


def create_list_helper(tree: HuffmanTree, all_bytes: list) -> None:
    """
    Helper method that creates a list of values, depending on the
    representation of Huffman Tree <tree>
    """
    if tree is None:  # = not internal node
        pass
    elif tree.left is None and tree.right is None:  # = not internal node
        pass
    else:  # = internal node
        create_list_helper(tree.left, all_bytes)  # traverse lhs
        create_list_helper(tree.right, all_bytes)  # traverse rhs
        if tree.left is not None or tree.right is not None:  # traverse the root
            if tree.left.left is None and tree.left.right is None:
                # the lhs is a leaf, so add 0 and tree.left.symbol
                all_bytes.append(0)
                all_bytes.append(tree.left.symbol)
            else:
                # lhs is not a leaf, so add 1 and tree.left.number
                all_bytes.append(1)
                all_bytes.append(tree.left.number)
            # do the same with the rhs
            if tree.right.left is None and tree.right.right is None:
                # (rhs is a leaf b/c right.left and right.right is None)
                all_bytes.append(0)
                all_bytes.append(tree.right.symbol)
            else:
                all_bytes.append(1)
                all_bytes.append(tree.right.number)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree) + int32_to_bytes(
        len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # l_type: 0/1 (if the corresponding HuffmanTree's left is a leaf)
    # l_data: a symbol or the node number of a HuffmanTree's left
    # r_type: 0/1 (if the corresponding HuffmanTree's right is a leaf)
    # r_data: a symbol or the node number of a HuffmanTree's right

    left = None
    right = None

    if node_lst[root_index].l_type == 0:  # 0: indicates it's a leaf
        left = HuffmanTree(node_lst[root_index].l_data)
    if node_lst[root_index].r_type == 0:  # 0: indicates it's a leaf
        right = HuffmanTree(node_lst[root_index].r_data)
    if node_lst[root_index].l_type == 1:  # 1: recurse on the lhs
        left = generate_tree_general(node_lst, node_lst[root_index].l_data)
    if node_lst[root_index].r_type == 1:  # 1: recurse on the rhs
        right = generate_tree_general(node_lst, node_lst[root_index].r_data)

    return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst,2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    left = None
    right = None

    if node_lst[root_index].l_type == 0:  # 0 indicates it's a leaf
        left = HuffmanTree(node_lst[root_index].l_data)
    if node_lst[root_index].r_type == 0:  # 0 indicates it's a leaf
        right = HuffmanTree(node_lst[root_index].r_data)
    # postorder: right child -> left child -> root
    #          x            -> node_lst[root_index]
    #      A       x
    #   B              C
    # here, B is node_lst[root_index]-len(node_lst)
    # B (left child) is 1 position left of C, so subtract 1
    if node_lst[root_index].l_type == 1:  # recurse on the node using helper
        # left child is len(node_lst) positions left of the root
        left = generate_tree_general(node_lst,
                                     node_lst[root_index].l_data - len(
                                         node_lst))
    if node_lst[root_index].r_type == 1:  # recurse on the node using helper
        # right child is len(node_`lst) - 1 positions left of the left child
        right = generate_tree_general(node_lst, node_lst[root_index].r_data - (
            len(node_lst) - 1))
    return HuffmanTree(None, left, right)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    curr = tree
    output = []
    bits = ''
    # iterate over the bits in <bytes> extracted from byte_to_bits
    for byte in text:
        bits += byte_to_bits(byte)

    for bit in bits:
        # traverse the tree using the current bit
        if bit == '0':
            curr = curr.left
        if bit == '1':
            curr = curr.right

        # if leaf is reached, (yay) store symbol in <output>
        if curr.left is None and curr.right is None:
            output.append(curr.symbol)
            curr = tree  # reset
        else:
            pass
        if len(output) == size:
            return bytes(output)  # <size> bits is reached, so return

    return bytes(output)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, wit vhout changing its
     shape,by swapping nodes. The improvements are with respect to the
     dictionary of symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # sorting freq_dict from lowest to highest frequency/value
    # used _ instead of value:
    # https://dbader.org/blog/meaning-of-underscores-in-python
    sorted_keys = []
    for key, _ in sorted(freq_dict.items(), key=sort_by_value):
        sorted_keys.append(key)

    # use a helper method to correct tree.symbol
    swaps(tree=tree, freq_dict=sorted_keys)


def sort_by_value(key_value_pair: dict) -> int:
    """
    helper method
    """
    return key_value_pair[1]


def swaps(tree: HuffmanTree, freq_dict: dict, key_index: int = 0) -> None:
    """
    steps to swap:
    1. sort frequency dictionary (by increasing frequency)
    2. make a new dictionary of the current huffman tree
    3. traverse through new dictionary and make the symbol of
        every leaf the frequency dictioary key instead
    4. if its not a leaf re-add until a leaf is reached
    [idk if i'm supposed to swap everything]
    """
    # swap <node_dict> with <freq_dict> one value at a time
    node_dict = {1: [tree]}
    while len(node_dict[1]) > 0:
        # remove a node from node_dict (the incorrect order)
        # remove a node from freq_dict (the correct order)
        # swap node_dict.symbol with freq_dict.symbol to fix
        curr = node_dict[1].pop()
        if curr.left is None and curr.right is None:
            # replace symbol with what it SHOULD be
            curr.symbol = freq_dict[key_index]
            key_index += 1
        else:
            # re-add the lhs and rhs into node_list until leaf is reached
            if curr.left is not None:
                node_dict[1].append(curr.left)
            if curr.right is not None:
                node_dict[1].append(curr.right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
