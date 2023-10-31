import torch
import math


def main():
    input = torch.tensor([1000, 100, 40, 45, 6500, 100, 200,
                         21, 1083, 4875, 22, 0, 3, 4]).view(2,-1)
    input = input / input.max()
    print(input)
    # import ipdb; ipdb.set_trace()
    output = positional_encoder(input)
    t_input = positional_decoder(output, dimention=2)
    print(t_input)

def positional_encoder(input, dimention=10):
    sensor_list = []
    for d in range(dimention):
        L = 2**d
        sensor_list += [torch.sin(input*math.pi*L), torch.cos(input*math.pi*L)]
    output = torch.stack(sensor_list)
    return output

def positional_decoder(input, dimention=10):
    a = input[:,0,0]
    # import ipdb; ipdb.set_trace()
    range_x = [-1.0,1.0]
    for d in range(dimention):
        sin = math.asin(a[d*2])
        cos = math.acos(a[d*2+1])
        if d==dimention-1:
            x = math.atan2(a[d**2], a[d**2+1])
            output = range_x[0] + x/(2**d)
            import ipdb; ipdb.set_trace()
        if sin > 0:
            if cos > 0:
                range_x = [range_x[0]*0.5 + range_x[1]*0.5, range_x[0]*0.25 + range_x[1]*0.75]
            else:
                range_x = [range_x[0]*0.25 + range_x[1]*0.75, range_x[1]]
        else:
            if cos > 0:
                range_x = [range_x[0]*0.75 + range_x[1]*0.25, range_x[0]*0.5 + range_x[1]*0.5]
            else:
                range_x = [range_x[0], range_x[0]*0.75 + range_x[1]*0.5]
            
        # if d==0:
        #     continue
        # # x = math.atan2(a[2*d], a[2*d+1])
        # x = math.asin(a[2*d])
        # if x < 0:
        #     range_x[0] = sum(range_x)
        # else:
        #     range_x[1] = sum(range_x)
        # if d == (dimention - 1):
        #     x = range_x / (math.pi*2**(d-1))
        #     output = range_x[0] + x
    return output
        
            

    pass

if __name__=="__main__":
    main()