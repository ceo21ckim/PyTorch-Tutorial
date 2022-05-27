import time 

def spend_time(start_time, end_time):
    spend = end_time - start_time 
    spend_min = spend // 60 
    spend_sec = spend - spend_min *60 
    return spend_min, spend_sec

def accuracy(pred_y, true_y):
    pred_y = pred_y.argmax(dim = 1)
    true_y = pred_y.argmax(dim = 1)
    acc = (pred_y == true_y).sum()
    return acc 