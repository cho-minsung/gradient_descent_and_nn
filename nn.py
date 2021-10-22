import numpy
import matplotlib.pyplot as plt


class Example:
    def __init__(self):

        testx = numpy.array([1, 2])
        testw = numpy.array([[0.1, 0.2], [0.3, 0.4]])
        testw2 = numpy.array([[0.2], [1], [-3]])
        testw3 = numpy.array([[1], [2]])

        s1 = testx.dot(testw)

        print(s1)

        x1 = numpy.insert(s1, 0, 1)

        print(x1)

        activated = self.tansh(x1)

        print(activated)

        s2 = activated.dot(testw2)

        print(s2)

        activated = self.tansh(s2)

        print(activated)

        s3 = activated.dot(testw3)

        print(s3)

        activated = numpy.tanh(s3)

        print(activated)

    def tansh(in_x):
        if len(in_x) != 1:
            theta = numpy.tanh(in_x[1:])
        else:
            theta = numpy.tanh(in_x)
        theta = numpy.insert(theta, 0, 1)

        return theta


def activation_function(x_in):
    if len(x_in) != 1:
        theta = 1 / (1 + numpy.exp(-1 * x_in[1:]))
    else:
        theta = 1 / (1 + numpy.exp(-1 * x_in))
    theta = numpy.insert(theta, 0, 1)

    return theta


def derivative_of_activation(out):
    der = out * (1 - out)
    return der


def forward_propagation(input_x, expected_output, first_weight, second_weight):
    s1 = input_x.dot(first_weight)
    # print("s1: %s" % s1)

    x1 = numpy.insert(s1, 0, 1)

    x1 = activation_function(x1)

    # print("x1: %s" % x1)

    s2 = x1.dot(second_weight)

    # print("s2: %s" % s2)

    x2 = numpy.insert(s2, 0, 1)

    x2 = activation_function(x2)

    trained_out = x2[1:]

    print("y: %s" % trained_out)
    print("expected y: %s" % expected_output)

    return s1, s2, x1[1:], trained_out


def back_propagation(forward_outcome, expected_outcome, previous_result, previous_weight, learning_rate, recording):
    error = numpy.sum(0.5 * numpy.square((numpy.subtract(expected_outcome, forward_outcome))))
    recording.append(error)
    print("error: %s" % error)

    # back propagation

    der_error_out = -1 * numpy.subtract(expected_outcome, forward_outcome)

    # print("derivative of total error per output: %s" % der_error_out)

    der_out_net = derivative_of_activation(expected_outcome)

    # print("derivative of output per net: %s" % der_out_net)

    der_net_weight = previous_result

    # print("derivative of net input per weight: %s" % der_net_weight)

    der_error_weight = der_error_out * der_out_net * der_net_weight

    # print("derivative of error per weight: %s" % der_error_weight)

    correction = learning_rate * der_error_weight

    adjusted_weight = numpy.subtract(previous_weight, correction)

    return adjusted_weight, der_error_out, der_out_net, der_net_weight

def back_propagation_2(der_error_out, der_out_net, previous_weight, der_net_weight, weight_to_change, initial_values, learning_rate):
    der_error_net = der_error_out * der_out_net
    # print("der_E_net: %s" % der_error_net)
    der_e_out = der_error_net * previous_weight
    der_etotal_out = numpy.sum(der_e_out)
    # print("derivative of total error per output: %s" % der_etotal_out)

    der_out_net = der_net_weight * (1 - der_net_weight)
    der_etotal_weight = der_out_net * initial_values
    updated_weights = weight_to_change - learning_rate * der_etotal_weight
    return updated_weights


xi = numpy.array([1, 0.05, 0.10])
yi = numpy.array([0.01, 0.99])

w1 = numpy.array([[0.35, 0.35], [0.15, 0.25], [0.20, 0.30]])
w2 = numpy.array([[0.60, 0.60], [0.40, 0.50], [0.45, 0.55]])

learning_rate = 0.01

recorded_error = []

for i in range(20000):
    print("run #%d" % i)
    first_s, second_s, first_dot, outcome = forward_propagation(xi, yi, w1, w2)
    w2, der_error_out, der_out_net, der_net_weight = back_propagation(outcome, yi, second_s, w2, learning_rate, recorded_error)
    # print("adjusted w2: %s" % w2)
    w1 = back_propagation_2(der_error_out, der_out_net, w2, der_net_weight, w1, xi[1:], learning_rate)
    # print("adjusted w1: %s" % w1)


print("final test %s:" % outcome)
plt.plot(recorded_error)
plt.show()