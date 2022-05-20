import numpy as np
from fenics import *

def fem_solve(a, N, kappa, Q, gD, gN=None):
    x0, y0, x1, y1 = -a, -a, a, a

    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), N, N)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = kappa*inner(grad(u), grad(v))*dx 

    if not gN is None:
        # def boundary(x):
        #     return x[1] < y0 + DOLFIN_EPS or x[1] > y1 - DOLFIN_EPS
        def boundary(x):
            return x[1] < y0 + 1e-10 or x[1] > y1 - 1e-10

        L = Constant(0)*v*dx + Constant(gN)*v*ds
        bc = DirichletBC(V, Constant(gD), boundary)
        A, b = assemble_system(a, L, bc)
    else:
        L = Constant(0)*v*dx
        bc = DirichletBC(V, Constant(gD), DomainBoundary())
        
        A, b = assemble_system(a, L, bc)

    delta = PointSource(V, Point(0.0, 0.0,), Q)
    delta.apply(b)

    u = Function(V)
    solve(A, u.vector(), b)

    U = u.compute_vertex_values().reshape(N+1, N+1)
    XY = mesh.coordinates().reshape(N+1, N+1, 2)
    X, Y = XY[:, :, 0], XY[:, :, 1]
    return np.array((X, Y, U))

if __name__ == '__main__':

    # caps = np.array([ 0.5555,1.1111, 1.6666, 1.8888]) * -10000
    # caps = np.array([ 5555, 8888, 11111, 13333]) * -1
    # caps = np.arange(-15000, -5000, 1000)
    caps = np.arange(0.2, 2.4, 0.2)*-1
    # weight = np.array([[0.25, 0.25], [0.25, 0.25]])
    for cap in caps:
        X, Y, U = fem_solve(a=1, N=399, kappa=1, Q=cap, gD=0)
        name = '/home/whujjq/pinn/fd/real/green/' + f'{cap:.2f}' + '.npy'
        np.save(name, U)

    # print(X[0,:])
    # fig = plt.figure(figsize=(12, 6))

    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # surf = ax.plot_surface(X, Y, U,
    #                     cmap=cm.rainbow, linewidth=0, antialiased=False)
    # cax = fig.add_axes([0.1, 0.2, 0.01, 0.6])
    # fig.colorbar(surf, cax=cax)
    # np.save('/home/whujjq/pinn/fv/u/real_1.npy',U)
    # ax = fig.add_subplot(1, 2, 2)
    # # cs = ax.contour(X, Y, U,
    # #                 colors='k', levels=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    
    # cs = ax.contour(X, Y, U,
    #                 colors='k', levels=[-0.4, -0.3, -0.2, -0.15, -0.1, -0.05])
    # ax.clabel(cs, fmt='%.2f', inline=1)
    # ax.set_aspect('equal', adjustable='box')
    # plt.show()