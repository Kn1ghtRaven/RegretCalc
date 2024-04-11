import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from tqdm import tqdm
from scipy.optimize import minimize_scalar

def make_all_r(start=1, end=2, step_size=0.01, dec=2):
    """a function that returns an array with all the possible stopping times for the adv"""
    optimal_times = np.arange(start, end, step_size)
    optimal_times = np.append(optimal_times, end)
    optimal_times = np.round(optimal_times, decimals=dec)
    return optimal_times


def channel1_cap(p):
    """
    A function that returns the MI you would get from channel 1
    :param p: the value P
    :return: the MI you would get from channel 1
    """
    return 1-p


def channel2_cap(p):
    """
    A function that returns the MI you would get from channel 2
    :param p: the value P
    :return: the MI you would get from channel 2
    """
    global channel2_div
    return p/channel2_div



def time_until_ones(r):
    """
    this funcion gets the time it takes for the adv to finish and
    returns the time where we would get 1's for the first time
    """
    return r-calc_alpha(r)


def calc_alpha(r):
    """

    param r: the time the adv will finish the game 
    return: the number of 1's we will have until time r
    """
    global channel2_div
    return (1-r)/(channel2_div-1) +1 

def calc_gain(channel, p, time):
    """

    :param channel: what is the channel that you corrently trasmit on
    :param p:
    :param time: for how long are you going to stay with the same channel and P
    :return: the MI you would gain from this time
    """
    if channel == 1:
        return channel1_cap(p) * time
    else:
        return channel2_cap(p) * time


def time_to_finish(p, trasmited):
    """
    A function that returns the time you need to finish with probability p under channel 2
    """
    return (1 - trasmited) / channel2_cap(p)

def calc_regret(player_strats, optimal_times):
    """
    A function that returns the Regret you will have from a set of P and swich times
    :param player_strats: a list of tupels where the first element is the swich_time until what time you stay at p and the second is p
    :param optimal_times: a list of all the options the adv can finish the game in
    :return: the worst regret you can have with both
    """
    global channel2_div
    equality_prob = channel2_div / (channel2_div+1)
    regret = -np.inf
    worst_r = 1
    for optimal_time in optimal_times:
        time = 0
        trasmited = 0
        twos = time_until_ones(optimal_time)
        finished = False
        for (swich_time, p) in player_strats:
            if twos > swich_time:
                trasmited += calc_gain(2, p, swich_time - time)
                time = swich_time
                continue
            elif swich_time > twos and time < twos:
                trasmited += calc_gain(2, p, twos - time)
                time = twos
            if swich_time < optimal_time:
                trasmited += calc_gain(1, p, swich_time - time)
                time = swich_time
                continue
            elif swich_time >= optimal_time and time < optimal_time:
                trasmited += calc_gain(1, p, optimal_time - time)
                time = optimal_time
            ttf = time_to_finish(p, trasmited)
            if swich_time < ttf + time:
                trasmited += calc_gain(2, p, swich_time - time)
                time = swich_time
                continue
            else:
                if regret < time + ttf - optimal_time:
                    regret = time + ttf - optimal_time
                    worst_r = optimal_time
                    finished = True
        if not finished:
            ttf = time_to_finish(equality_prob, trasmited)
            if regret < time + ttf - optimal_time:
                regret = time + ttf - optimal_time
                worst_r = optimal_time
    return regret, worst_r


def calc_comp_ratio(player_strats, optimal_times):
    """
    A function that returns the Regret you will have from a set of P and swich times
    :param player_strats: a list of tupels where the first element is the swich_time until what time you stay at p and the second is p
    :param optimal_times: a list of all the options the adv can finish the game in
    :return: the worst regret you can have with both
    """
    comp_ratio = 1
    worst_r = 1
    for optimal_time in optimal_times:
        time = 0
        trasmited = 0
        twos = time_until_ones(optimal_time)
        finished = False
        for (swich_time, p) in player_strats:
            if twos > swich_time:
                trasmited += calc_gain(2, p, swich_time - time)
                time = swich_time
                continue
            elif swich_time > twos and time < twos:
                trasmited += calc_gain(2, p, twos - time)
                time = twos
            if swich_time < optimal_time:
                trasmited += calc_gain(1, p, swich_time - time)
                time = swich_time
                continue
            elif swich_time >= optimal_time and time < optimal_time:
                trasmited += calc_gain(1, p, optimal_time - time)
                time = optimal_time
            ttf = time_to_finish(p, trasmited)
            if swich_time < ttf:
                trasmited += calc_gain(2, p, swich_time - time)
                time = swich_time
                continue
            else:
                if comp_ratio > (optimal_time/(time + ttf) ):
                    comp_ratio = ( optimal_time/(time + ttf))
                    worst_r = optimal_time
                    finished = True
        if not finished:
            ttf = time_to_finish(2 / 3, trasmited)
            if comp_ratio > ( optimal_time/(time + ttf) ):
                comp_ratio = (optimal_time/(time + ttf) )
                worst_r = optimal_time
    return comp_ratio, worst_r

def generate_prob_arrays(size, start, end, step, divide_by=1000, end_value = 2/3):
    """

    :param size: the number of probabilities you will have
    :param start: the lowest probability in the array
    :param end: the highest probability
    :param step: the is the step size you will take when you increse one item in the array
    :param divide_by: value to enter in order to get all the values to be fructions
    :param end_value: the value of the end as a fruction
    :return: array of arrays of all the possible options
    """
    if size <= 0:
        return []

    def backtrack(curr_array, index, divide_by=1000):
        if index == size:
            arrays.append(curr_array.copy())
            return
        for num in range(start, end + 1, step):
            if not curr_array or num > curr_array[-1]:
                if num + step > end:
                    curr_array.append(end_value)
                else:
                    if len(curr_array) >= 1 and num / divide_by < curr_array[-1]:
                        continue
                    curr_array.append(num / divide_by)
                backtrack(curr_array, index + 1)
                curr_array.pop()

    arrays = []
    backtrack([], 0, divide_by)
    return arrays


def generate_time_arrays(swiches, start=100, end=200, step=1, divide_by=100):
    """

    :param swiches: the number of probabilities -1
    :param start: The lowest time in the array
    :param end: The highest time
    :param step: Is the step size you will take when you increse one item in the array
    :param divide_by: Value to enter in order to get all the values to be fructions
    :return: array of arrays of all the possible options for time swiches
    """
    if swiches <= 0:
        return []

    def backtrack(curr_array, index, divide_by=100):
        if index == swiches:
            finished_arr = curr_array.copy()
            finished_arr.append(end/divide_by)
            arrays.append(finished_arr.copy())
            return
        for num in range(start, end + 1, step):
            if not curr_array or num > curr_array[-1]:
                if num + step > end:
                    curr_array.append(end/divide_by)
                else:
                    if len(curr_array) >= 1 and num / divide_by < curr_array[-1]:
                        continue
                    curr_array.append(num / divide_by)
                backtrack(curr_array, index + 1)
                curr_array.pop()

    arrays = []
    backtrack([], 0, divide_by)
    return arrays


def find_best_prob(all_probs, swich_times):
    """
    A function that gets swich_times and an array of arrays of probabilities and it finds the best one
    :param all_probs: An array of arrays of probabilities you want to check
    :param swich_times: An array of time swiches
    :return: The array of the best probabilities the regret and the r that gets this regret.
    """
    best_regret = np.inf
    best_probabilities = []
    meching_r = 1
    for probs in all_probs:
        player_strats = np.array(list(zip(swich_times, probs)))
        regret, worst_r = calc_regret(player_strats, optimal_times)
        if regret < best_regret:
            best_probabilities = probs.copy()
            best_regret = regret
            meching_r = worst_r
    return best_probabilities, best_regret, meching_r


def find_best_prob_cr(all_probs, swich_times):
    """
    A function that gets swich_times and an array of arrays of probabilities and it finds the best one
    :param all_probs: An array of arrays of probabilities you want to check
    :param swich_times: An array of time swiches
    :return: The array of the best probabilities the regret and the r that gets this regret.
    """
    best_cr = 1
    best_probabilities = []
    meching_r = 1
    for probs in all_probs:
        player_strats = np.array(list(zip(swich_times, probs)))
        comp_ratio, worst_r = calc_comp_ratio(player_strats, optimal_times)
        if comp_ratio < best_cr:
            best_probabilities = probs.copy()
            best_cr = comp_ratio
            meching_r = worst_r
    return best_probabilities, best_cr, meching_r


def find_best_time_prob(all_probs,all_times):
    """
    A function that finds the Regret
    :param all_probs:  An array of arrays of probabilities you want to check
    :param all_times:  An array of arrays of time swiches you want to check
    :return: The array of the best probabilities the regret and the r that gets this regret and the array of time swiches.
    """
    best_swiches = []
    best_regret_time = np.inf
    best_probabilities_time = []
    maching_r_time = 1
    for swich_times in tqdm(all_times):
        best_probabilities, best_regret, maching_r = find_best_prob(all_probs,swich_times)
        if best_regret_time > best_regret :
            best_regret_time = best_regret
            best_swiches = swich_times.copy()
            maching_r_time  = maching_r
            best_probabilities_time = best_probabilities.copy()
    print(f"best regret : {best_regret_time}")
    print(f"meching r : {maching_r_time}")
    print(f"best_probabilities: {best_probabilities_time}")
    print(f"swich_times : {best_swiches}")
    return best_probabilities_time, best_regret_time, maching_r_time, best_swiches


def find_best_time_prob_cr(all_probs,all_times):
    """
    A function that finds the Regret
    :param all_probs:  An array of arrays of probabilities you want to check
    :param all_times:  An array of arrays of time swiches you want to check
    :return: The array of the best probabilities the regret and the r that gets this regret and the array of time swiches.
    """
    best_swiches = []
    best_cr_time = 1
    best_probabilities_time = []
    maching_r_time = 1
    for swich_times in tqdm(all_times):
        best_probabilities, best_cr, maching_r = find_best_prob_cr(all_probs,swich_times)
        if best_cr_time > best_cr:
            best_cr_time = best_cr
            best_swiches = swich_times.copy()
            maching_r_time  = maching_r
            best_probabilities_time = best_probabilities.copy()
    print(f"best CR : {best_cr_time}")
    print(f"meching r : {maching_r_time}")
    print(f"best_probabilities: {best_probabilities_time}")
    print(f"swich_times : {best_swiches}")
    return best_probabilities_time, best_cr_time, maching_r_time, best_swiches


def store_variable(variable, filename):
    """this functions saves a value in a pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(variable, f)


def load_variable(filename):
    """this function loads a var from a pickle file"""
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    return variable

def load_results(best_probabilities_time, best_regret_time, maching_r, best_swiches):
    best_probabilities_time = load_variable('best_probabilities_time.pkl')
    print("best_probabilities_time :", best_probabilities_time)
    best_regret_time = load_variable('best_regret_time.pkl')
    print("best_regret_time :", best_regret_time)
    maching_r_time = load_variable('maching_r_time.pkl')
    print("maching_r_time :", maching_r_time)
    best_swiches = load_variable('best_swiches.pkl')
    print("best_swiches :", best_swiches)

def save_results(best_probabilities_time, best_regret_time, maching_r, best_swiches):
    store_variable(best_probabilities_time,"best_probabilities_time.pkl")
    store_variable(best_regret_time, "best_regret_time.pkl")
    store_variable(maching_r, "maching_r_time.pkl")
    store_variable(best_swiches, "best_swiches.pkl")

def generate_arrays(start, my_range, step, mid, end, dec = 3):
    result = []
    first, second = sorted(my_range)
    x = 0
    while start + x * step < second:
        result.append([start, np.round_(first + x * step, decimals=dec), mid, end])
        x += 1
    return result
    

def generate_arrays_p2( my_range, step, end, dec = 3):
    result = []
    first, second = sorted(my_range)
    x = 0
    while first + x * step < second:
        result.append([np.round_(first + x * step, decimals=dec), end])
        x += 1
    return result

def test_reg2(optimal_times):
    swich_times = [1.45,2]
    probs = [0.505,2/3]
    player_strats = np.array(list(zip(swich_times, probs)))
    regret, worst_r = calc_regret(player_strats, optimal_times)
    print(f"regret: {regret} , worst_r : {worst_r}")


if __name__ == '__main__':
    channel2_div = 5
    equality_prob = channel2_div / (channel2_div+1)
    optimal_times = make_all_r(start=1, end=channel2_div, step_size=0.001, dec=3)
    size = 2
    divide_by_prob = 1000
    start_value_prob = 1 / 2 * divide_by_prob
    end_value_prob = equality_prob * divide_by_prob
    step = 0.005 * divide_by_prob
    divide_by_time = 100 
    start_time = 100 
    end_time = channel2_div * divide_by_time
    step_time = 5
    # swich_times = [1, 1.25, 1.5, 2]  # need to add one more for the last dist
    # swich_times = generate_arrays(1,(1.1,1.4),1.5,2)
    all_probs = generate_prob_arrays(size, int(np.round_(start_value_prob)), int(np.round_(end_value_prob)), int(np.round_(step)), divide_by_prob,equality_prob)
    all_times = generate_time_arrays(size - 1, start=start_time, end=end_time, step=step_time, divide_by=divide_by_time)
    # all_times = generate_arrays_p2((1.99,2.01),0.01,3, 2)
    print(len(all_probs))
    print("all times : ")
    print(all_times)   
    print(f"the size of all_times is : {len(all_times)}")

    #Regret 
    start = time.time()
    best_probabilities_time, best_regret_time, maching_r_time, best_swiches = find_best_time_prob(all_probs, all_times)
    end = time.time()
    print(f"Execution time of regret{size} is : {end - start}" )

    #CR not working at the moment :(
    # start = time.time()
    # best_probabilities_time, best_regret_time, maching_r_time, best_swiches = find_best_time_prob_cr(all_probs, all_times)
    # end = time.time()
    # print(f"Execution time of comp ratio{size} is : {end - start}")

    # save result
    store_variable(best_probabilities_time,f"best_probabilities_time_{channel2_div}_{size}.pkl")
    store_variable(best_regret_time, f"best_regret_time_{channel2_div}_{size}.pkl")
    store_variable(maching_r_time, f"maching_r_time_{channel2_div}_{size}.pkl")
    store_variable(best_swiches, f"best_swiches_{channel2_div}_{size}.pkl")

    # load previous result
    best_probabilities_time = load_variable(f"best_probabilities_time_{channel2_div}_{size}.pkl")
    print("best_probabilities_time :", best_probabilities_time)
    best_regret_time = load_variable(f"best_regret_time_{channel2_div}_{size}.pkl")
    print("best_regret_time :", best_regret_time)
    maching_r_time = load_variable(f"maching_r_time_{channel2_div}_{size}.pkl")
    print("maching_r_time :", maching_r_time)
    best_swiches = load_variable(f"best_swiches_{channel2_div}_{size}.pkl")
    print("best_swiches :", best_swiches)


