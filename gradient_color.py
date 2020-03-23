def RGB_to_hex(rgb):
    rgb = [int(x) for x in rgb]
    return "#" + "".join([
        "0{0:x}".format(v) if v < 16 else
        "{0:x}".format(v) for v in rgb])


def hex_to_RGB(hex):
    return [int(hex[i:i + 2], 16) for i in range(1, 6, 2)]


def linear_gradient(start_hex, finish_hex='#FFFFFF', n=10):
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    rgb_list = [RGB_to_hex(s)]
    for t in range(1, n):
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)]
        rgb_list.append(RGB_to_hex(curr_vector))

    return rgb_list
