# /usr/local/bin/python

from __future__ import division
import os
import math
import numpy as np
import json
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline


from pymatgen.io.vasp.sets import Poscar
from pymatgen.io.vasp.outputs import Oszicar
from pymatgen.core import Species, Lattice, Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.sites import PeriodicSite
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.bond_valence import calculate_bv_sum
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries


__author__ = 'saigautam,pcanepa'
__copyright__ = "Copyright 2015 superMg"
__version__ = "0.5"
__maintainer__ = "both of us"
__email__ = "gautam91@mit.edu,pcanepa@lbl.gov"
__date__ = "Oct 22 , 2015"

"""
Notes: The get_img_percent_length() has been cleaned up. Works fine on structures where the atom crosses the boundary in all
       3-directions
       Always use with care and check end results. Please get back if there are any unusual errors.
"""


def checkint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]


def dirlist(root_path):
    """
    Returns a list of directories containing all the images and the start and end points of NEB calculations.
    root_path = Path to the directory containing the results of the NEB run.
    Please make sure you have the naming sequence for the images as XY where X and Y are integers. Let 00 represent the
    starting point and the last integer (09 for example in a system of 8 images) represent the end point.
    """
    dirs = listdirs(root_path)
    dirs_list = []
    for i in range(0, len(dirs)):
        if checkint(dirs[i]) and len(dirs[i]) == 2:
            dirs_list.append(dirs[i])
    return dirs_list


def sort_dirs(dirslist):
    dirs_new = []
    for i in dirslist:
        dirs_new.append(int(i))
        dirs_new.sort()
    for j in range(0, len(dirslist)):
        if len(str(dirs_new[j])) == 1:
            dirslist[j] = '0'+str(dirs_new[j])
        elif len(str(dirs_new[j])) == 2:
            dirslist[j] = str(dirs_new[j])
        else:
            assert 0 == 1, "Check directory numbering or naming. It should be of the form XY such that XY is an integer"
    return dirslist


def checkstart(dirslist, root_path):
    """
    Checks if there is a directory numbered 00 and has a POSCAR inside
    """
    if '00' in dirslist and os.path.isfile(os.path.join(root_path, '00/POSCAR')) and os.path.isfile(os.path.join(root_path, '00/OSZICAR')):
        return True
    else:
        return False


def checkend(dirslist, root_path):
    """
    Known bug: Both CONTCARS and POSCARS are present in images. Check may not be rigorous.
    """
    if os.path.isfile(os.path.join(root_path, dirslist[-1]+'/POSCAR')) and os.path.isfile(os.path.join(root_path, dirslist[-1]+'/OSZICAR')):
        return True
    else:
        return False


def get_all_poscars(dirslist, root_path):
    poscar = []
    if checkstart(dirslist, root_path) and checkend(dirslist, root_path):
        for i in range(0, len(dirslist)):
            tmpath = os.path.join(root_path, dirslist[i])
            if i != 0 and i != len(dirslist)-1:
                path = os.path.join(tmpath, "CONTCAR")
            else:
                path = os.path.join(tmpath, "POSCAR")
            poscar.append(Poscar.from_file(path))

    else:
        assert 0 == 1, "Check your start and end directories. POSCAR may be missing"
    return poscar


def get_all_oszicars(dirslist, root_path):
    oszicar = []
    # print(root_path)
    if checkstart(dirslist, root_path) and checkend(dirslist, root_path):
        for i in range(0, len(dirslist)):
            path = os.path.join(root_path, dirslist[i], "OSZICAR")
            oszicar.append(Oszicar(path))
    else:
        assert 0 == 1, "Check your start and end directories. OSZICAR may be missing"
    return oszicar


def get_cart_images(dirslist):
    """
    Gets cartesian coordinates of all images in the NEB as a list.
    dirlist= List of directories that contain the POSCARs or the CONTCARs
    """
    cart_all = []
    poscar = get_all_poscars(dirslist, root_path)
    for i in range(0, len(dirslist)):
        cart_all.append(poscar[i].structure.lattice.get_cartesian_coords(
            poscar[i].structure.frac_coords))
    return cart_all


def get_frac_images(dirslist):
    """
    Gets fractional coordinates of all images in the NEB as a list.
    dirlist= List of directories that contain the POSCARs or the CONTCARs
    """
    frac_all = []
    poscar = get_all_poscars(dirslist, root_path)
    for i in range(0, len(dirslist)):
        frac_all.append(poscar[i].structure.frac_coords)
    return frac_all


def get_cart_from_frac(coords, lattice):
    """
    Gets cartesian coordinates for a set of fractional coordinates and lattice matrix passed.
    Very trivial function. Ensure that the order of fractional coordinates and the lattice match. No checks made.
    """
    cart_coords = []
    for i in range(0, len(coords)):
        cart_coords.append(
            coords[i]*(lattice[0][i]+lattice[1][i]+lattice[2][i]))
    return cart_coords


def get_energies_images(dirslist):
    energies = []
    oszicar = get_all_oszicars(dirslist, root_path)
    for i in range(0, len(dirslist)):
        energies.append(oszicar[i].ionic_steps[-1]["E0"])
    return energies


def get_atommap(dirslist, poscars_list):
    atommap = []
    if checkstart(dirslist, root_path):
        poscar = poscars_list[0]
        no_atoms = poscar.natoms
        element = poscar.site_symbols
        count = 1
        for k in range(0, len(element)):
            if k != 0:
                count += int(no_atoms[k-1])
            for l in range(0, int(no_atoms[k])):
                atommap.append(element[k]+str(l+count))
    else:
        assert 0 == 1, "There is no directory number 00. Check your directory lists"
    return atommap


def get_diff_sp(poscars, atommap, user_sp='', user_pos=None):
    """
    Returns diffusing species and it's position
    Known bug: Assumes the diffusing species is dilute, i.e., it's the kind of element that's present in least
    quantity in the POSCAR. This may not be a rigorous check. Please pass in user_sp if results are contradicting.
    Might be difficult to track the diffusing species in dilute vacancy limit.
    user_sp: If you already know the diffusing species or want to fix a particular element, pass it here.
    user_pos: If there are multiple diffusing ions in the lattice, use this variable to choose one for which
            the bond distances should be calculated. Pass an integer, string or a single level list. One can
            extract user_pos from the dictionary of user_pos and atommap
    """
    poscar = poscars[0]
    no_atoms = poscar.natoms
    dict_atoms = {}
    element = poscar.site_symbols
    for i in range(0, len(element)):
        dict_atoms.update({element[i]: int(no_atoms[i])})
        no_atoms[i] = int(no_atoms[i])
    no_atoms.sort()
    if user_sp != '':
        sp = user_sp
    elif no_atoms[0] != no_atoms[1] and no_atoms[0] == 1:
        sp = dict_atoms.keys()[list(dict_atoms.values()).index(no_atoms[0])]
    else:
        assert 0 == 1, "There is at least 1 more element in the unit cell with the same number of atoms as the diffusing \
                     species. Specify user_sp and user_pos if this is the case."

    pos = []
    if user_pos is not None:
        if type(user_pos) is list:
            posn = user_pos
            for d in posn:
                pos.append(int(d))
        elif checkint(user_pos):
            pos.append(int(user_pos))
        else:
            assert 0 == 1, "user_pos must be an int, str or list"
    else:
        for j in atommap:
            s = ''
            posn = s
            for k in range(0, len(j)):
                if not checkint(j[k]):
                    s += j[k]
                else:
                    posn += j[k]
            if sp == s:
                pos.append(int(posn))
    if sp+str(pos[0]) not in atommap:
        assert 0 == 1, "Values supplied in user_sp and user_pos dont match Poscar values. Check input"
    else:
        diff_sp = {sp: pos}
    return diff_sp


def get_lattice_constant(poscars):
    """
    Function to get lattice constants. Uses pymatgen.core.lattice; returns a list of [x,y,z]
    """
    if type(poscars) is list:
        poscar = poscars[0]
    else:
        poscar = poscars
    return poscar.structure.lattice.abc


def get_lattice(poscars):
    """
    Function to get lattice constant matrix. Returns a 2d list if one POSCAR is passed. If a list of
    POSCARs are passed, then returns a 3D-list
    """
    if type(poscars) is not list:
        return poscars.structure.lattice._matrix
    else:
        matrix = []
        for i in poscars:
            matrix.append(i.structure.lattice._matrix)
        return matrix


def dict_userpos_atommap(atommap):
    """
    Write a dictionary to convert atommap->user_pos to make sure that the maps are error free.
    """
    atommap_userpos = {}
    s = ''
    for i in atommap:
        for j in range(0, len[i]):
            if checkint(i[j]):
                s += i[j]
        pos = int(s)
        atommap_userpos.update({i: pos})
    return atommap_userpos


def get_Voronoi(poscars, diff_sp):
    """
    Known bug, solid_angle_tol = 0.5 works for Mn2O4 spinels. Needs to be checked for others.
    solid_angle_tol is the tolerance on the solid angle, bigger the angle- closer the atoms.
    """
    solid_angle_tol = 0.5
    sites = []
    n = int(list(diff_sp.values())[0][0])-1
    for i in range(0, len(poscars)):
        CNN = CrystalNN(weighted_cn=True)
        voronoi_nn = CNN.get_nn_info(poscars[i].structure, n)
        # Here I'm using a minimum weight of 0.5 to filter out nearest neighbors that are too far away (weights are always between 0 and 1)
        # In general, undistorted polyhedra will have sites with weights near 1
        image_sites = []
        for nn in voronoi_nn:
            if nn['weight'] > 0.5:
                image_sites.append(nn['site'])
        sites.append(image_sites)
    return sites


def get_cn(poscars, diff_sp):
    """
    A function to compute the coordinations numbers not rounded 
    Output: cn
    """
    n = int(list(diff_sp.values())[0][0])-1
    cn = []
    for i in range(0, len(poscars)):
        CNN = CrystalNN(weighted_cn=True)
        coord_no = CNN.get_cn(poscars[i].structure, n, use_weights=True)
        cn.append(coord_no)
    return cn


def get_nn_bond_length(poscars, neighbors, diff_sp):
    """
    Returns a list of bond lengths required for bond valence method on each image.
    Alert : 2D list is returned
    """
    bond_lengths = []
    n = int(list(diff_sp.values())[0][0])-1
    for j in range(0, len(poscars)):
        dist = []
        for i in range(0, len(neighbors[j])):
            sp = int(neighbors[j][i])-1
            dist.append(poscars[j].structure.get_distance(n, sp))
        bond_lengths.append(dist)
    return bond_lengths


def get_neighbors(poscars, atommap, voronoi_sites, tol=1e-6):
    """
    Gets the indices of the neighboring atoms near the diffusing species (or any other species whose voronoi_sites
    are known.
    The indices are returned as a list for each poscar passed.
    The indices have the same numerical value as atommap/user_pos
    """
    neighbors = []
    for l in range(0, len(poscars)):
        array = []
        count = 'False'
        for k in range(0, len(voronoi_sites[l])):
            for j in range(0, len(atommap)):
                for i in range(0, 3):
                    if (abs(voronoi_sites[l][k].frac_coords[i]-poscars[l].structure.frac_coords[j][i]) < tol):
                        count = 'True'
                    else:
                        if(abs(voronoi_sites[l][k].frac_coords[i]-(poscars[l].structure.frac_coords[j][i]-1.0)) < tol):
                            poscars[l].structure.frac_coords[j][i] -= 1.0
                            count = 'True'
                        elif(abs(voronoi_sites[l][k].frac_coords[i]-(poscars[l].structure.frac_coords[j][i]+1.0)) < tol):
                            poscars[l].structure.frac_coords[j][i] += 1.0
                            count = 'True'
                        else:
                            count = 'False'
                            break
                if count == 'True':
                    array.append(int(j)+1)
        neighbors.append(array)
    return neighbors


def bond_valence_sum(poscars, neighbors, diff_sp, bond_lengths, gga_scale_factor=1.015):
    """
    A function to compute bond valence
    Output: Bond Valence sum normalized by the coordination number
    """
    bv_sum = []
    n = int(list(diff_sp.values())[0][0])-1
    for j in range(0, len(poscars)):
        site_dist = []
        for i in range(0, len(neighbors[j])):
            # -1 to adjust for numbering scheme of indices
            anion_sites = poscars[j].structure.sites[int(neighbors[j][i])-1]
            temp_list = (anion_sites, bond_lengths[j][i])
            site_dist.append(temp_list)
        CNN = CrystalNN(weighted_cn=True)
        coord_no = CNN.get_cn(poscars[j].structure, n, use_weights=True)
        cn = []
        for i in range(0, len(poscars)):
            cn.append(float(len(bond_lengths[i])))
        # print bond_lengths[i]
        # print len(bond_lengths[i])
        bv_sum.append(calculate_bv_sum(
            poscars[j].structure.sites[n], site_dist, gga_scale_factor)/cn[j])
        # bv_sum.append(calculate_bv_sum(poscars[j].structure.sites[n],site_dist,gga_scale_factor)/coord_no.get_coordination_number(n))
    return bv_sum


def get_coordination_environment(poscars, diff_sp):
    """
    Output: Coordination environment of the moving atom along the path
    """

    n = int(list(diff_sp.values())[0][0])-1
    acg = AllCoordinationGeometries()
    coord_dict = acg.get_symbol_name_mapping()
    ces = []

    for poscar in poscars:

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(centering_type='centroid', include_central_site_in_centroid=True,
                             structure_refinement=lgf.STRUCTURE_REFINEMENT_NONE)
        lgf.setup_structure(structure=poscar.structure)
        se = lgf.compute_structure_environments(maximum_distance_factor=1.41)
        simplest_strategy = SimplestChemenvStrategy(
            distance_cutoff=1.4, angle_cutoff=0.3)
        lse = LightStructureEnvironments.from_structure_environments(
            strategy=simplest_strategy, structure_environments=se)

        if lse.coordination_environments[n]:
            ces.append(
                coord_dict[lse.coordination_environments[n][0]['ce_symbol']])
        else:
            ces.append(None)

    return ces


def plotter(x, y, label_axis, plot_title, filename_out, xrange_interp=None, fit=False, cnt_Last=False):
    """
    A function to plot results on the fly
    Uses matplotlib
    xrange_interp: if interpolation is required
    fit: if linear fitting is required
    cnt_Last: avoid python to hold on the first plot plotted
    Output: encapsulated postscript
    """
    # plot attributes
    label_font = {'style': 'normal', 'weight': 'normal',
                  'stretch': 'normal', 'size': 32}
    fig, ax1 = pyplot.subplots()
    ax1.plot(x, y, 'o', ms=10, mfc='red')

    ax1.set_xlabel(label_axis[0], label_font)
    # ax1.set_xticklabels(np.linspace(0,10),fontsize=20,family='Helvetica')
    ax1.xaxis.set_tick_params(labelsize=48)

    ax1.set_ylabel(label_axis[0], label_font)
    ax1.yaxis.set_tick_params(labelsize=48)

    pyplot.title(plot_title, fontsize=48)

    if xrange_interp != None and fit == False:
        xnew = np.linspace(0, xrange_interp, 1000)
        # cubic interpolation
        f = interp1d(x, y, kind='cubic')
        ax1.plot(xnew, f(xnew), '-', lw=2)

    if fit:
        # linear fit
        xnew = np.linspace(np.min(x), np.amax(x), 1000)
        f = np.poly1d(np.polyfit(x, y, 1))
        ax1.plot(xnew, f(xnew), '-', lw=2)
        # pyplot.legend()

    ax1.annotate('Max ' + str(np.amax(y)) + ' meV', xy=(2, 1), xytext=(35, 100), fontsize=48, horizontalalignment='left',
                 verticalalignment='bottom')

    pyplot.savefig(os.path.join(root_path, filename_out),
                   format='eps', dpi=1000)
    pyplot.tight_layout()
    pyplot.draw()
    if cnt_Last:
        pyplot.show()


def setup_plot():

    # Infinite plotting stuff from down. Might not require cleaning.
    plt.rcParams['text.latex.preamble'] = [
        # i need upright \micro symbols, but you need...
        r'\usepackage{siunitx}',
        # ...this to force siunitx to actually use your fonts
        r'\usepackage[T1]{fontenc}',
        r'\sisetup{detect-all}',
        r'\usepackage{helvet}',  # set the normal font here
        # load up the sansmath so that math -> helvet##
        r'\usepackage[eulergreek,EULERGREEK]{sansmath}',
        r'\sansmath',  # <- tricky! -- gotta actually tell tex to use!
    ]

    plt.rcParams['mathtext.fallback_to_cm'] = 'True'
    #plt.rc('font', family='sans-serif')
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.rc('text', usetex=True)
    plt.rcParams['lines.antialiased'] = True

    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.loc'] = 'upper center'
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.fontsize'] = '24'
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams['legend.scatterpoints'] = 1
    plt.rcParams['legend.edgecolor'] = 'inherit'

    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 1.
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['axes.linewidth'] = 1.


def write_file(filename, array):
    dataout = open(os.path.join(root_path, filename), 'w')
    dataout.write(array)
    dataout.close()


def get_img_percent_length(poscars, all_frac_list, diff_sp, lattice_constant_tolerance=0.75):
    """
    Returns a list of piece-by-piece distance between adjacent images.
    Requires a list of poscar objects, the list of fractional coordinates and the diffusing species.
    The lattice constant tolerance is to check whether a translation in coordinates is required for an ion moving across
    a periodic boundary.
        For example, the check is such that if
        Distance(x_coordinate(image1),x_coordinate(image2)) >= (lattice_constant_tolerance * x_lattice_constant),
        then the image is considered to move across a boundary
    """
    img_percent_length = []
    lattice = get_lattice(poscars[0])
    d = int(list(diff_sp.items())[0][1][0])-1

    # Start_End_positions
    start_frac = all_frac_list[0][d]
    end_frac = all_frac_list[-1][d]
    start = get_cart_from_frac(start_frac, lattice)
    end = get_cart_from_frac(end_frac, lattice)
    path_length = math.sqrt(
        (end[0]-start[0])**2+(end[1]-start[1])**2+(end[2]-start[2])**2)
    # Ideal piece-wise length for linear interpolation
    tol = path_length/float(len(all_frac_list)-2)
    lat_tol = lattice_constant_tolerance
    # Scaled lattice constants
    lat = [lat_tol*x for x in get_lattice_constant(poscars[0])]
    print_flag = True

    # Looping through Images
    for k in range(0, (len(all_frac_list)-1)):
        frac = all_frac_list[k][d]
        frac1 = all_frac_list[k+1][d]
        cart = get_cart_from_frac(frac, lattice)
        cart1 = get_cart_from_frac(frac1, lattice)
        dist = math.sqrt((cart[0]-cart1[0])**2 +
                         (cart[1]-cart1[1])**2+(cart[2]-cart1[2])**2)
        if dist <= tol:
            img_percent_length.append(dist)
        else:
            x_contrib = abs(cart[0]-cart1[0])
            y_contrib = abs(cart[1]-cart1[1])
            z_contrib = abs(cart[2]-cart1[2])
            if (abs(cart[0]-cart1[0]) >= lat[0]):
                x_contrib = abs(cart[0]-cart1[0]) - \
                    abs(lattice[0][0]+lattice[1][0]+lattice[2][0])
            if (abs(cart[1]-cart1[1]) >= lat[1]):
                y_contrib = abs(cart[1]-cart1[1]) - \
                    abs(lattice[0][1]+lattice[1][1]+lattice[2][1])
            if (abs(cart[2]-cart1[2]) >= lat[2]):
                z_contrib = abs(cart[2]-cart1[2]) - \
                    abs(lattice[0][2]+lattice[1][2]+lattice[2][2])
            new_dist = math.sqrt(x_contrib**2+y_contrib**2+z_contrib**2)
            if new_dist >= tol:
                if print_flag:
                    print("""
                    Looks like the diffusing ion is moving in across a periodic boundary.
                    Ensure the results look ok.
                    Else, try changing the lattice constant tolerance
                    """)
                    print_flag = False
                img_percent_length.append(dist)
            else:
                img_percent_length.append(new_dist)
    return img_percent_length


def get_actual_diff_sp(poscars, all_cart_list, atommap, user_sp=' ', user_pos=None):
    """
    Caveat: Returns the atom that moves the longest. It shouldn't matter in coupled diffusion. But if you do want two
    different path lengths, pass the user_sp and user_pos seperately
    In case of doubt, always pass the user_sp and user_pos explicitly
    """
    if user_sp == ' ' and user_pos is None:
        prev_dist = 0.0
        prev_posn = 0
        prev_s = ' '
        for j in atommap:
            s = ''
            posn = s
            for k in range(0, len(j)):
                if not checkint(j[k]):
                    s += j[k]
                else:
                    posn += j[k]
            posn = int(posn)
            cart = all_cart_list[0][posn-1]
            cart1 = all_cart_list[-1][posn-1]
            dist = math.sqrt((cart[0]-cart1[0])**2 +
                             (cart[1]-cart1[1])**2+(cart[2]-cart1[2])**2)
            if dist > prev_dist:
                prev_dist = dist
                prev_posn = posn
                prev_s = s
    else:
        prev_s = user_sp
        prev_posn = user_pos
    diff_sp = get_diff_sp(poscars, atommap, prev_s, prev_posn)
    return diff_sp


def get_path_length(img_percent_length):
    """
    Input: List of 'piece-wise' diffusing atom distances between images
    Returns total diffusion length (in Angstroms)
    Should be identical to path_length defined in get_img_percent_length(), serves as decent cross-check
    """
    path_length = 0
    for i in img_percent_length:
        path_length += i
    return path_length


def get_percentage_path_length(img_percent_length):
    path_length = get_path_length(img_percent_length)
    count = 0
    img_percent_length = np.asarray(img_percent_length)
    for i in range(0, len(img_percent_length)):
        img_percent_length[i] += count
        if i != 0:
            count += img_percent_length[i] - img_percent_length[i-1]
        elif i == 0:
            count += img_percent_length[0]
    percentage_length = np.insert(img_percent_length, 0, 0)
    percentage_length = (percentage_length/path_length)*100
    return percentage_length


def get_scaled_energies(energies_images):
   # energy_endpoint_mean = (energies_images[0] + energies_images[-1])*0.5
    energy_endpoint_mean =  energies_images[0]
    scaled_energies = np.asarray(energies_images)
    scaled_energies = (scaled_energies-energy_endpoint_mean)*1000
    return scaled_energies


def get_effective_cn(bond_lengths):
    ecn = []
    for i in bond_lengths:
        i.sort()
        lmin = i[0]
        num = 0.0
        den = 0.0
        for j in i:
            exp = (math.exp(1-(j/lmin)**6))
            num += j*exp
            den += exp
        lav = num/den
        weight = 0.0
        for j in i:
            weight += math.exp(1-(j/lav)**6)
        ecn.append(weight)
    return ecn


def get_distortion_index(bond_lengths):
    di = []
    for i in bond_lengths:
        i.sort()
        lmin = i[0]
        num = 0.0
        den = 0.0
        for j in i:
            exp = (math.exp(1-(j/lmin)**6))
            num += j*exp
            den += exp
        lav = num/den
        DI = 0.0
        for j in i:
            DI += abs(j-lav)/lav
        di.append(DI/len(i))
    return di


########################################################################################################################

# Sample code using the above functions and writing the required outputs. Please do needed changes here alone.
parser = argparse.ArgumentParser()

parser.add_argument('-ld', '--list-of-directories',
                    help='<Required> -ld path1 path2 path3 ...',
                    required=True,
                    dest='subdirectories',
                    default=["path1", "path2", "path3"],
                    nargs='+'
                    )

args = parser.parse_args()
subdirectories = args.subdirectories

'''
fix labels for plotting 
'''
lab = []
for item in subdirectories:
    if "_" in item:
        lab.append(item.replace("_U", "+$U$"))
    else:
        lab.append(item)
labels=[]
for item in lab:
    labels.append(item.replace("/",""))
symbols = ['o', 's', '^', 'p']
barriers_max = []
x_spline = np.arange(0, 100, 0.1)

setup_plot()
fig, ax1 = plt.subplots()
fig.subplots_adjust(hspace=0.2, wspace=0.4)

#fig.subplots_adjust(hspace=0.2, wspace=0.4)


#colors = sns.color_palette("pastel")
colors = sns.color_palette("colorblind")
for i in colors:
    print (i)
#colors = sns.color_palette("Set2")
#colors =sns.color_palette("Set2")

'''
SHIFTS colors  if GGA_U is not there 
'''
if "GGA_U" not in subdirectories:
    colors = [colors[0], colors[1], colors[3], colors[4]]
    
print(colors)
print('Start Start Start Start Start ')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Summary')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Sub-directories found:', subdirectories)
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


for i, item in enumerate(subdirectories):
    root_path = os.path.join(os.getcwd(), item)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print("Working directory:", root_path)

    dirs_num = dirlist(root_path)
    dirs_num = sort_dirs(dirs_num)
    print('NEB directories', dirs_num)
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    poscars = get_all_poscars(dirs_num, os.path.join(os.getcwd(), item))
    all_cart_list = get_cart_images(dirs_num)
    all_frac_list = get_frac_images(dirs_num)
    atommap = get_atommap(dirs_num, poscars)
    diff_sp = get_actual_diff_sp(poscars, all_cart_list, atommap)

    latt_const = get_lattice_constant(poscars)
    oszicars = get_all_oszicars(dirs_num, root_path)
    energies_images = get_energies_images(dirs_num)

    sites = get_Voronoi(poscars, diff_sp)
    array = get_neighbors(poscars, atommap, sites)
    cn = get_cn(poscars, diff_sp)
    bond_lengths = get_nn_bond_length(poscars, array, diff_sp)
    #ces = get_coordination_environment(poscars, diff_sp)
    ces = False

    # From Vesta and Voronoi. Not rigorous Vesta
    effective_cn = get_effective_cn(bond_lengths)
    distortion_index = get_distortion_index(bond_lengths)  # Ditto

    #bv_sum = bond_valence_sum(poscars,array,diff_sp,bond_lengths)

    img_percent_lengthances = get_img_percent_length(
        poscars, all_frac_list, diff_sp)
    path_length = get_path_length(img_percent_lengthances)
    img_percent_length = get_percentage_path_length(img_percent_lengthances)
    energies_scaled = get_scaled_energies(energies_images)
    get_effective_cn(bond_lengths)

    bonds_t = []
    for z in range(0, len(poscars)):
        bonds_t.append(sum(bond_lengths[z][:])/len(bond_lengths[z][:]))

    cn = []
    for z in range(0, len(poscars)):
        cn.append(len(bond_lengths[z][:]))

    final_array = np.vstack(
        (img_percent_length, energies_scaled, cn, bonds_t,  effective_cn, distortion_index))

    df = pd.DataFrame((np.transpose(final_array)), columns=[
                      'path',  'E',  'CN', 'BL', 'ECN', 'DI'])
    df.to_csv(os.path.join(root_path, "data.csv"))

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Analysis')
    print("Path length:", path_length)
    print('Saving summary CSV file:', os.path.join(root_path, "data.csv"))
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    cubic_e = UnivariateSpline(df['path'], df['E'])
    cubic_e.set_smoothing_factor(0.8)
    y_spline = cubic_e(x_spline)
    min_val = np.amin(y_spline)
    print("MAX", df['E'].max())
    ax1.plot(df['path'], df['E'], marker=symbols[i], color=colors[i],
             label=labels[i]+ " " + str(df['E'].round(0).max()), lw=1, ls=(0, (5, 5)))

#ls=(0, (1, 5))

    #ax1.plot(x_spline, y_spline, '-', color=colors[i])
    barriers_max.append(df['E'].round(0).max())


#x_bar = np.arange(len(barriers_max))
#barplot = ax2.bar(x_bar, barriers_max, color=colors, width=1/3)


"""
for i in [ax1, ax2]:
    i.grid(False)
    if i==ax1:
         i.tick_params(axis='both', labelsize=16)
    else:
        i.tick_params(axis='x', labelsize=12)
        i.tick_params(axis='y', labelsize=16)

    i.set_ylabel(r'Migration energy (meV)', size=18)
    i.tick_params(direction='in', which="major", length=7, width=1.5)
    i.tick_params(direction='in', which="minor", length=5, width=1)


"""

ax1.tick_params(direction='in', which="major", length=7, width=1.5)
ax1.tick_params(direction='in', which="minor", length=0.5, width=0.1)
ax1.set_ylabel(r'Migration energy (meV)', size=18)
ax1.grid(False)
ax1.tick_params(axis='both', labelsize=16)

ax1.tick_params(axis='both', which='minor', labelsize=16)


 ### ax1.tick_params(axis='both', labelsize=16)


"""
ax2.set_xticks(x_bar)
ax2.set_xticklabels(labels)
ax2.xaxis.set_ticks_position('none')
"""
ax1.xaxis.set_major_locator(MultipleLocator(20))
ax1.xaxis.set_minor_locator(MultipleLocator(10))


ax1.set_xlabel(r'Path distance (\%)', size=18)
#ax2.set_xlabel(r'XC functional', size=18)
#ax2.tick_params(axis='x', labelsize=20)
ax1.grid(axis='both', linestyle = 'dotted', linewidth = 0.6 )

#ax1.legend(loc='right', bbox_to_anchor=(0.25,0.8), prop={'size': 10}, ncol=1).get_frame().set_linewidth(1.)

#ax1.legend(loc='upper center',bbox_to_anchor=(1.5,1.1), prop={'size': 10}, ncol=1).get_frame().set_linewidth(1.)
ax1.set_ylim([-400.0, max(barriers_max)+100.0])


"""
ax1.set_yticklabels('')
ax1.set_yticks([110, 310, 510, 710, 910, 1110, 1310 ],   minor=True)
ax1.set_yticklabels(['0','200','400','600','800', '1000', '1200'], minor=True)
"""
#max(barriers_max)+100.0

#ax2.set_ylim([min(barriers_max)-50.0, max(barriers_max)+100.0])


#x_bar = np.arange(len(barriers_max))
#barplot = ax2.bar(x_bar, barriers_max, color=colors, width=1/3)
 


"""
for item in barplot:
    height = item.get_height()
    ax2.annotate(
        '{:4d}'.format(int(height)),
        xy=(item.get_x() + 0.5*item.get_width(), height),
        xytext=(2,3),  # 3 points vertical offset
        textcoords="offset points",
        ha='center',
        va='bottom',
        rotation=45,
        size=14
    )

"""

plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "barriers.pdf"),
            format="pdf",bbox_inches='tight')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Now plotting all barriers')
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Labels for plotting:')
for i, item in enumerate(subdirectories):
    print(i, labels[i])
print('Plot saved in:', os.path.join(os.getcwd(), "barriers.pdf"))
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('End End End End End End End')
