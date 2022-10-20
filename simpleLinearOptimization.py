###############################################################################
##
##  Purpose: Solve a simple linear optimization problem on your own
##
##  minimize 1*x1 + 2*x2
##  such that
##           x1 + x2 = 1,
##  and
##           x1, x2 >= 0
##
##  Note: x1 = 1 - x2 => min(1-x2 + 2x2) => min(1 + x2) with x2 > 0 => (x1,x2) = (1,0) = 1
##
###############################################################################

from mosek.fusion import *

def main():

    A = [[1.0,1.0]]

    c = [1.0, 2.0]

    # create the model according to mosek specification

    with Model("simpleLO") as M:
        # Create a variable 'x' of length 2
        x = M.variable("x", 2, Domain.greaterThan(0.0))

        # Create constraints
        #M.constraint("c1", Expr.dot(A[0], x), Domain.equalsTo(1.0))
        M.constraint("c1", Expr.sum(x), Domain.equalsTo(1.0))


        # Set the objective function
        M.objective("obj", ObjectiveSense.Maximize, Expr.dot(c, x))

        M.solve()

        sol = x.level()
        print('\n'.join(["x[%d] = %f" % (i, sol[i]) for i in range(2)]))

if __name__ == '__main__':
    main()
