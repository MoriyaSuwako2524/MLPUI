

def calculate_parameters(state_dict, prefix=""):
    params = 0
    for k in state_dict.keys():
        if k.startswith(prefix):
            w = state_dict[k]
            params += w.nelement()
    return params