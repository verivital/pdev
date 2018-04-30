'''
This module implements printers to print discrete-space models (ODE) of PDE benchmarks to SpaceEx and Flow* format
Dung Tran: 4/26/2018
'''
import numpy as np
from scipy.optimize import linprog


def get_dynamics(C, symbol):
    'print y = C * symbol, symbol = x -> dynamics = state dynamics, symbol = u -> dynamics = input dynamics'

    # example y = Cx

    assert isinstance(C, np.ndarray)
    assert isinstance(symbol, str)
    m = C.shape[0]
    n = C.shape[1]

    dynamics = []

    for i in xrange(0, m):
        yi = ''
        for j in xrange(0, n):
            if C[i, j] > 0:
                cx = '{}*{}{}'.format(C[i, j], symbol, j)
                if j == 0:
                    yi = '{} {}'.format(yi, cx)
                else:
                    if yi != '':
                        yi = '{} + {}'.format(yi, cx)
                    else:
                        yi = '{}'.format(cx)
            elif C[i, j] < 0:
                cx = '{}*{}{}'.format(-C[i, j], symbol, j)
                yi = '{} - {}'.format(yi, cx)
        if yi == '':
            yi = '0'
        dynamics.append(yi)

    return dynamics


def print_spaceex_xml_file_autonomous_ode(A, C, file_name):
    'print spaceex xml file for dot{x} = Ax, y = Cx'

    assert isinstance(
        A, np.ndarray) and A.shape[0] == A.shape[1], 'error: A is not an ndarray or A is not a square matrix'
    assert isinstance(C, np.ndarray), 'error: C is not an ndarray'
    assert C.shape[1] == A.shape[0], 'error: inconsistent between A and C'
    assert isinstance(file_name, str), 'error: file name should be a string'

    n = A.shape[0]
    m = C.shape[0]
    xml_file = open(file_name, 'w')

    xml_file.write('<?xml version="1.0" encoding="iso-8859-1"?>\n')
    xml_file.write(
        '<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">\n')

    # print core component
    xml_file.write('  <component id="core_component">\n')

    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))
    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const"/>\n')

    xml_file.write(
        '    <location id="1" name="Model" x="362.0" y="430.0" width="426.0" height="610.0">\n')

    # print invariant
    xml_file.write('      <invariant>\n')
    xml_file.write('        t &lt;= stoptime\n')
    output_dynamics = get_dynamics(C, 'x')
    for i in xrange(0, m):
        xml_file.write(
            '        &amp;y{} == {}\n'.format(
                i, output_dynamics[i]))
    xml_file.write('      </invariant>\n')

    # print flow
    xml_file.write('      <flow>\n')
    xml_file.write('        t\' == 1\n')
    for i in xrange(0, n):
        C1 = A[i]
        xi_dynamics = get_dynamics(C1, 'x')
        xml_file.write('        &amp;x{}\' == {}\n'.format(i, xi_dynamics[0]))

    xml_file.write('      </flow>\n')
    xml_file.write('    </location>\n')
    xml_file.write('  </component>\n')

    # print system
    xml_file.write('  <component id="sys">\n')
    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const" controlled="true"/>\n')

    xml_file.write('    <bind component="core_component" as="model">\n')
    for i in xrange(0, n):
        xml_file.write('      <map key="x{}">x{}</map>\n'.format(i, i))
    for i in xrange(0, m):
        xml_file.write('      <map key="y{}">y{}</map>\n'.format(i, i))
    xml_file.write('      <map key="t">t</map>\n')
    xml_file.write('      <map key="stoptime">stoptime</map>\n')
    xml_file.write('    </bind>\n')

    xml_file.write('  </component>')

    xml_file.write('</sspaceex>')

    xml_file.close()

    return xml_file


def print_spaceex_cfg_file_autonomous_ode(
        xmin_vec, xmax_vec, ymin_vec, ymax_vec, stoptime, step, file_name):
    'print configuration file for spaceex model of autonomous ODE: dot{x} = Ax, y = Cx'

    assert isinstance(xmin_vec, list), 'error: xmin_vec is not a list'
    assert isinstance(xmax_vec, list), 'error: xmax_vec is not a list'
    assert len(xmin_vec) == len(
        xmax_vec), 'error: inconsistency between xmin_vec and xmax_vec'

    assert isinstance(ymin_vec, list), 'error: ymin_vec is not a list'
    assert isinstance(ymax_vec, list), 'error: ymax_vec is not a list'
    assert len(ymin_vec) == len(
        ymax_vec), 'error: inconsistency between ymin_vec and ymax_vec'
    assert isinstance(file_name, str)

    cfg_file = open(file_name, 'w')

    cfg_file.write('# analysis option \n')
    cfg_file.write('system = "sys"\n')

    # init string
    init_str = ''
    n = len(xmin_vec)
    m = len(ymin_vec)
    for i in xrange(0, n):
        init_str = '{} x{} >= {} & x{} <= {} &'.format(
            init_str, i, xmin_vec[i], i, xmax_vec[i])
    for i in xrange(0, m):
        init_str = '{} y{} >= {} & y{} <= {} &'.format(
            init_str, i, ymin_vec[i], i, ymax_vec[i])
    init_str = '{} t == 0 & stoptime == {}'.format(init_str, stoptime)

    cfg_file.write('initially = "{}"\n'.format(init_str))
    cfg_file.write('scenario = "supp" \n')
    cfg_file.write('directions = "box"\n')
    cfg_file.write('sampling-time = {}\n'.format(step))
    cfg_file.write('time-horizon = {}\n'.format(stoptime))
    cfg_file.write('iter-max = 10\n')
    output = ''
    for i in xrange(0, m):
        if i < m - 1:
            output = '{}y{}, '.format(output, i)
        else:
            output = '{}y{}'.format(output, i)
    cfg_file.write('output-variables = "t, {}" \n'.format(output))
    cfg_file.write('output-format = "GEN"\n')
    cfg_file.write('rel-err = 1.0e-8\n')
    cfg_file.write('abs-err = 1.0e-12\n')

    cfg_file.close()

    return cfg_file


def print_spaceex_xml_file_non_autonomous_ode(A, B, C, file_name):
    'print spaceex xml file for dot{x} = Ax + Bu, y = Cx'

    assert isinstance(
        A, np.ndarray) and A.shape[0] == A.shape[1], 'error: A is not an ndarray or A is not a square matrix'
    assert isinstance(C, np.ndarray), 'error: C is not an ndarray'
    assert isinstance(B, np.ndarray), 'error: B is not an ndarray'
    assert C.shape[1] == A.shape[0] == B.shape[0], 'error: inconsistent between A, B and C'
    assert isinstance(file_name, str), 'error: file name should be a string'

    n = A.shape[0]    # number of state variables
    m = C.shape[0]    # number of outputs
    q = B.shape[1]    # number of inputs u

    xml_file = open(file_name, 'w')

    xml_file.write('<?xml version="1.0" encoding="iso-8859-1"?>\n')
    xml_file.write(
        '<sspaceex xmlns="http://www-verimag.imag.fr/xml-namespaces/sspaceex" version="0.2" math="SpaceEx">\n')

    # print core component
    xml_file.write('  <component id="core_component">\n')

    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))

    for i in xrange(0, q):
        xml_file.write(
            '    <param name="u{}" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n'.format(i))

    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const"/>\n')

    xml_file.write(
        '    <location id="1" name="Model" x="362.0" y="430.0" width="426.0" height="610.0">\n')

    # print invariant
    xml_file.write('      <invariant>\n')
    xml_file.write('        t &lt;= stoptime\n')
    output_dynamics = get_dynamics(C, 'x')
    for i in xrange(0, m):
        xml_file.write(
            '        &amp;y{} == {}\n'.format(
                i, output_dynamics[i]))
    xml_file.write('      </invariant>\n')

    # print flow
    xml_file.write('      <flow>\n')
    xml_file.write('        t\' == 1\n')
    for i in xrange(0, q):
        xml_file.write('         &amp;u{}\' == 0\n'.format(i))
    for i in xrange(0, n):
        C1 = A[i]
        B1 = B[i]
        xi_dynamics = get_dynamics(C1, 'x')
        print "\nxi dynamics = {}".format(xi_dynamics)
        ui_dynamics = get_dynamics(B1, 'u')
        xml_file.write(
            '        &amp;x{}\' == {} + ({})\n'.format(i, xi_dynamics[0], ui_dynamics[0]))

    xml_file.write('      </flow>\n')
    xml_file.write('    </location>\n')
    xml_file.write('  </component>\n')

    # print system
    xml_file.write('  <component id="sys">\n')
    for i in xrange(0, n):
        xml_file.write(
            '    <param name="x{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    for i in xrange(0, m):
        xml_file.write(
            '    <param name="y{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    for i in xrange(0, q):
        xml_file.write(
            '    <param name="u{}" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n'.format(i))
    xml_file.write(
        '    <param name="t" type="real" local="false" d1="1" d2="1" dynamics="any" controlled="true"/>\n')
    xml_file.write(
        '    <param name="stoptime" type="real" local="false" d1="1" d2="1" dynamics="const" controlled="true"/>\n')

    xml_file.write('    <bind component="core_component" as="model">\n')
    for i in xrange(0, n):
        xml_file.write('      <map key="x{}">x{}</map>\n'.format(i, i))
    for i in xrange(0, m):
        xml_file.write('      <map key="y{}">y{}</map>\n'.format(i, i))
    for i in xrange(0, q):
        xml_file.write('      <map key="u{}">u{}</map>\n'.format(i, i))
    xml_file.write('      <map key="t">t</map>\n')
    xml_file.write('      <map key="stoptime">stoptime</map>\n')
    xml_file.write('    </bind>\n')

    xml_file.write('  </component>')

    xml_file.write('</sspaceex>')

    xml_file.close()

    return xml_file


def print_spaceex_cfg_file_non_autonomous_ode(
        xmin_vec, xmax_vec, ymin_vec, ymax_vec, umin_vec, umax_vec, stoptime, step, file_name):
    'print configuration file for spaceex model of non-autonomous ODE: dot{x} = Ax + Bu, y = Cx'

    assert isinstance(xmin_vec, list), 'error: xmin_vec is not a list'
    assert isinstance(xmax_vec, list), 'error: xmax_vec is not a list'
    assert len(xmin_vec) == len(
        xmax_vec), 'error: inconsistency between xmin_vec and xmax_vec'

    assert isinstance(ymin_vec, list), 'error: ymin_vec is not a list'
    assert isinstance(ymax_vec, list), 'error: ymax_vec is not a list'
    assert len(ymin_vec) == len(
        ymax_vec), 'error: inconsistency between ymin_vec and ymax_vec'

    assert isinstance(umin_vec, list), 'error: umin_vec is not a list'
    assert isinstance(umax_vec, list), 'error: umax_vec is not a list'
    assert len(umin_vec) == len(
        umax_vec), 'error: inconsistency between umin_vec and umax_vec'

    assert isinstance(file_name, str)

    cfg_file = open(file_name, 'w')

    cfg_file.write('# analysis option \n')
    cfg_file.write('system = "sys"\n')

    # init string
    init_str = ''
    n = len(xmin_vec)
    m = len(ymin_vec)
    q = len(umin_vec)

    for i in xrange(0, n):
        init_str = '{} x{} >= {} & x{} <= {} &'.format(
            init_str, i, xmin_vec[i], i, xmax_vec[i])
    for i in xrange(0, m):
        init_str = '{} y{} >= {} & y{} <= {} &'.format(
            init_str, i, ymin_vec[i], i, ymax_vec[i])
    for i in xrange(0, q):
        init_str = '{} u{} >= {} & u{} <= {} &'.format(
            init_str, i, umin_vec[i], i, umax_vec[i])
    init_str = '{} t == 0 & stoptime == {}'.format(init_str, stoptime)

    cfg_file.write('initially = "{}"\n'.format(init_str))
    cfg_file.write('scenario = "supp" \n')
    cfg_file.write('directions = "box"\n')
    cfg_file.write('sampling-time = {}\n'.format(step))
    cfg_file.write('time-horizon = {}\n'.format(stoptime))
    cfg_file.write('iter-max = 10\n')
    output = ''
    for i in xrange(0, m):
        if i < m - 1:
            output = '{}y{}, '.format(output, i)
        else:
            output = '{}y{}'.format(output, i)
    cfg_file.write('output-variables = "t, {}" \n'.format(output))
    cfg_file.write('output-format = "GEN"\n')
    cfg_file.write('rel-err = 1.0e-8\n')
    cfg_file.write('abs-err = 1.0e-12\n')

    cfg_file.close()

    return cfg_file


def get_ymin_ymax(C, xmin_vec, xmax_vec):
    'get ymin-max from the relation y = Cx'

    assert isinstance(C, np.ndarray), 'error: matrix C should be a numpy array'
    assert isinstance(xmin_vec, list), 'error: xmin_vec should be a list'
    assert isinstance(xmax_vec, list), 'error: xmax_vec should be a list'

    assert C.shape[1] == len(xmin_vec) == len(
        xmax_vec), 'error: inconsistent dimension'

    n = C.shape[1]
    m = C.shape[0]
    ymin_vec = []
    ymax_vec = []
    bounds = []

    for i in xrange(0, n):
        bounds.append((xmin_vec[i], xmax_vec[i]))

    print "\n length of bounds = {}".format(len(bounds))

    C1 = np.asarray(C)

    for i in xrange(0, m):
        c_vec = C1[i]

        opt_res_min = linprog(c=c_vec, bounds=bounds)
        opt_res_max = linprog(c=-c_vec, bounds=bounds)

        if opt_res_min.status == 0:
            ymin_vec.append(opt_res_min.fun)
        else:
            print "\nminimization error: can not find min value"

        if opt_res_max.status == 0:
            ymax_vec.append(-opt_res_max.fun)
        else:
            print "\nmaximization error: can not find max value"

    return ymin_vec, ymax_vec


class Printer(object):
    'Printer object'
    @staticmethod
    def print_spaceex_homo_odes(
            A, C, init_xmin_vec, init_xmax_vec, stoptime, step, file_name):
        'Print homogenous ODE dot{x} = Ax, y = Cx to SpaceEx format'

        # ODE format: dot{x} = Ax, y = Cx
        # init set : xmin[i] <= x[i] <= xmax[i]
        # the output is declared as invariant in spaceex model
        # To do that we need to compute the initial set of the outpu : ymin_vec[j] <= y[j] <= ymax[j]
        # This is done by using y = Cx

        assert isinstance(file_name, str)
        xml_file_name = '{}.xml'.format(file_name)
        cfg_file_name = '{}.cfg'.format(file_name)
        xml_file = print_spaceex_xml_file_autonomous_ode(
            A, C, xml_file_name)    # print xml file

        ymin_vec, ymax_vec = get_ymin_ymax(C, init_xmin_vec, init_xmax_vec)
        cfg_file = print_spaceex_cfg_file_autonomous_ode(
            init_xmin_vec, init_xmax_vec, ymin_vec, ymax_vec, stoptime, step, cfg_file_name)

        return xml_file, cfg_file

    @staticmethod
    def print_spaceex_non_homo_odes(
            A, B, C, init_xmin_vec, init_xmax_vec, umin_vec, umax_vec, stoptime, step, file_name):
        'Print homogenous ODE dot{x} = Ax + Bu, y = Cx to SpaceEx format'

        assert isinstance(file_name, str)
        xml_file_name = '{}.xml'.format(file_name)
        cfg_file_name = '{}.cfg'.format(file_name)
        xml_file = print_spaceex_xml_file_non_autonomous_ode(
            A, B, C, xml_file_name)    # print xml file

        ymin_vec, ymax_vec = get_ymin_ymax(C, init_xmin_vec, init_xmax_vec)
        cfg_file = print_spaceex_cfg_file_non_autonomous_ode(
            init_xmin_vec,
            init_xmax_vec,
            ymin_vec,
            ymax_vec,
            umin_vec,
            umax_vec,
            stoptime,
            step,
            cfg_file_name)

        return xml_file, cfg_file

    @staticmethod
    def print_flow_homo_odes(
            A, init_xmin_vec, init_xmax_vec, stoptime, step, file_name):
        'Print homogenous ODE dot{x} = Ax, to Flow* format'

        assert isinstance(
            A, np.ndarray) and A.shape[0] == A.shape[1], 'error: A is not an ndarray or A is not a square matrix'
        assert isinstance(init_xmin_vec, list) and isinstance(
            init_xmax_vec, list), 'error: xmin and xmax should be a list'
        assert isinstance(
            file_name, str), 'error: file name should be a string'
        assert A.shape[0] == len(init_xmin_vec) == len(
            init_xmax_vec), 'error: inconsistent dimensions between system matrix and initial vectors'

        n = A.shape[0]
        flow_model = open(file_name, 'w')

        flow_model.write('continuous reachability \n')
        flow_model.write('{ \n')

        # print state variables

        flow_model.write(' state var \n')
        var_str = ''
        for i in xrange(0, n):
            var_x = ' x{},'.format(i)
            var_str = var_str + var_x

        flow_model.write(var_str)
        flow_model.write(' t \n')

        # print setting

        flow_model.write(' setting \n')
        flow_model.write(' { \n')
        flow_model.write('  fixed steps {} \n'.format(step))
        flow_model.write('  time {} \n'.format(stoptime))
        flow_model.write('  remainder estimation 1e-3 \n')
        flow_model.write('  identity precondition \n')
        flow_model.write('  gnuplot octagon t, x0 \n')
        flow_model.write('  fixed order 30 \n')
        flow_model.write('  cutoff 1e-15 \n')
        flow_model.write('  precision 256 \n')
        flow_model.write('  output {} \n'.format(file_name))
        flow_model.write('  print on \n')
        flow_model.write(' } \n')

        # print lti ode

        flow_model.write(' lti ode \n')
        flow_model.write(' { \n')

        for i in xrange(0, n):
            C1 = A[i]
            xi_dynamics = get_dynamics(C1, 'x')
            flow_model.write('  x{}\' = {}\n'.format(i, xi_dynamics[0]))

        flow_model.write('  t\' = 1 \n')
        flow_model.write('  } \n')

        # print initial set

        flow_model.write(' init \n')
        flow_model.write(' { \n')

        for i in xrange(0, n):
            flow_model.write(
                '  x{} in [{}, {}] \n'.format(
                    i, init_xmin_vec[i], init_xmax_vec[i]))
        flow_model.write('  t in [0, 0] \n')
        flow_model.write(' } \n')

        # end model
        flow_model.write('}')

        return flow_model

    @staticmethod
    def print_flow_non_homo_odes(
            A, B, init_min_vec, init_max_vec, umin_vec, umax_vec, stoptime, step, file_name):
        'Print homogenous ODE dot{x} = Ax + Bu, to Flow* format'

        assert isinstance(
            A, np.ndarray) and A.shape[0] == A.shape[1], 'error: A is not an ndarray or not a square matrix'
        assert isinstance(B, np.ndarray), 'error: B is not an ndarray'

        assert B.shape[0] == A.shape[0], 'error: in consistency between A and B dimension'

        assert isinstance(
            init_max_vec, list), 'error: init_max_vec is not a list'
        assert isinstance(
            init_min_vec, list), 'error: init_min_vec is not a list'
        assert len(init_max_vec) == len(
            init_min_vec) == A.shape[0], 'error: inconsistency between initial condition and matrix A dimension'
        assert isinstance(umin_vec, list), 'error: umin_vec is not a list'
        assert isinstance(umax_vec, list), 'error: umax_vec is not a list'
        assert len(umin_vec) == len(
            umax_vec) == B.shape[1], 'error: inconsistency between input condition and matrix B dimension'

        n = A.shape[0]
        m = B.shape[1]

        flow_model = open(file_name, 'w')

        flow_model.write('continuous reachability \n')
        flow_model.write('{ \n')

        # print state variables

        flow_model.write(' state var \n')
        var_str = ''
        for i in xrange(0, n):
            var_x = ' x{},'.format(i)
            var_str = var_str + var_x
        flow_model.write(var_str)

        input_str = ''
        for i in xrange(0, m):
            input_u = ' u{},'.format(i)
            input_str = input_str + input_u
        flow_model.write(input_str)
        flow_model.write(' t \n')

        # print setting

        flow_model.write(' setting \n')
        flow_model.write(' { \n')
        flow_model.write('  fixed steps {} \n'.format(step))
        flow_model.write('  time {} \n'.format(stoptime))
        flow_model.write('  remainder estimation 1e-3 \n')
        flow_model.write('  identity precondition \n')
        flow_model.write('  gnuplot octagon t, x0 \n')
        flow_model.write('  fixed order 30 \n')
        flow_model.write('  cutoff 1e-15 \n')
        flow_model.write('  precision 256 \n')
        flow_model.write('  output {} \n'.format(file_name))
        flow_model.write('  print on \n')
        flow_model.write(' } \n')

        # print lti ode

        flow_model.write(' lti ode \n')
        flow_model.write(' { \n')

        for i in xrange(0, n):
            C1 = A[i]
            C2 = B[i]
            xi_dynamics = get_dynamics(C1, 'x')
            ui_dynamics = get_dynamics(C2, 'u')
            dynamics = '{} + ({})'.format(xi_dynamics[0], ui_dynamics[0])

            flow_model.write('  x{}\' = {}\n'.format(i, dynamics))

        flow_model.write('  t\' = 1 \n')
        flow_model.write('  } \n')

        # print initial set

        flow_model.write(' init \n')
        flow_model.write(' { \n')

        for i in xrange(0, n):
            flow_model.write(
                '  x{} in [{}, {}] \n'.format(
                    i, init_min_vec[i], init_max_vec[i]))
        for i in xrange(0, m):
            flow_model.write(
                '  u{} in [{}, {}] \n'.format(
                    i, umin_vec[i], umax_vec[i]))

        flow_model.write('  t in [0, 0] \n')
        flow_model.write(' } \n')

        # end model
        flow_model.write('}')

        return flow_model


def test():
    'test printer'

    A = np.matrix([[1, 1], [0, 1]])
    B = np.matrix([[1, 0], [1, 1]])
    C = np.matrix([1, 2])
    print "\nC = {}".format(C)
    print "\nshape C = {}".format(C.shape)

    xmin_vec = [1, 0]
    xmax_vec = [1.2, 1]
    umin_vec = [0.1, 0.2]
    umax_vec = [0.2, 0.3]

    stoptime = 1.0
    step = 0.1
    file_name = 'test'

    Printer().print_spaceex_non_homo_odes(A, B, C, xmin_vec,
                                          xmax_vec, umin_vec, umax_vec, stoptime, step, file_name)


def test_flow_printer():

    A = np.matrix([[1, 1], [0, 1]])
    B = np.matrix([[1, 0], [1, 1]])
    C = np.matrix([1, 2])
    print "\nC = {}".format(C)
    print "\nshape C = {}".format(C.shape)

    xmin_vec = [1, 0]
    xmax_vec = [1.2, 1]
    umin_vec = [0.1, 0.2]
    umax_vec = [0.2, 0.3]

    stoptime = 1.0
    step = 0.1
    file_name = 'test'

    Printer().print_flow_non_homo_odes(A, B, xmin_vec, xmax_vec, umin_vec, umax_vec, stoptime, step, file_name)


if __name__ == '__main__':
    # test()
    test_flow_printer()
