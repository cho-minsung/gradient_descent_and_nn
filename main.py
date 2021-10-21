import numpy
import sympy
import matplotlib.pyplot as plt

x, y, z = sympy.symbols('x y z')
f = x**2 + 2*y**2 + 2 * sympy.sin(2 * sympy.pi * x) * sympy.sin(2 * sympy.pi * y)
print("f(x, y) = %s" % f)
f_partial_x = sympy.diff(f, x)
print("df(x, y)/dx = %s" % f_partial_x)
f_partial_y = sympy.diff(f, y)
print("df(x, y)/dy = %s" % f_partial_y)


def gradient(plug_x, plug_y):
    out_x = sympy.N(f_partial_x.subs([(x, plug_x), (y, plug_y)]))
    out_y = sympy.N(f_partial_y.subs([(x, plug_x), (y, plug_y)]))
    return numpy.array([out_x, out_y])


x0 = -1  #theta1
y0 = -1  #theta2
initials = [x0, y0]
n = 0.01  #learning rate

function_output = sympy.N(f.subs([(x, x0), (y, y0)]))

def gradient_descent(f, gradient, initials, learning_rate, max_iteration):
    iteration = 0
    cost_record = numpy.empty(max_iteration)
    theta = initials.copy()

    while iteration < max_iteration:
        gradient_output = gradient(theta[0], theta[1])
        theta = theta - learning_rate * gradient_output
        function_output = sympy.N(f.subs([(x, theta[0]), (y, theta[1])]))
        errorx = gradient_output[0] - theta[0]
        errory = gradient_output[1] - theta[1]
        cost_record[iteration] = (errorx + errory)/2
        print("iteration %d:" % iteration)
        print("function output: %s" % function_output)
        print("gradient output: %s" % gradient_output)
        print("theta: %s" % theta)
        print("cost: %s" % cost_record[iteration])
        print("---------------------------------------")

        iteration += 1

    return theta, cost_record


cost_record = numpy.empty(50)

final_theta, cost_record = gradient_descent(f=f,
                               gradient=gradient,
                               initials=initials,
                               learning_rate=n,
                               max_iteration=50)

print("The function "+str(f)+" converges to a minimum")
print("Number of iterations: 50")
print("f(x, y) = %s" % function_output)
print("theta (x, y) = %s" % final_theta)
plt.plot(cost_record)
plt.title("Cost function vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()