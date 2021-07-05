
def add_two_dim_dict(adic, key_a, key_b, val): 
    if key_a in adic:
        adic[key_a].update({key_b: val})
    else:
        adic.update({key_a:{key_b: val}})

def add_three_dim_dict(adic, key_a, key_b, key_c, val):
    if key_a in adic:
        if key_b in adic[key_a]:
            adic[key_a][key_b].update({key_c: val})
        else:
            adic[key_a].update({key_b:{key_c: val}})
    else:
        adic.update({key_a: {key_b: {key_c: val}}})