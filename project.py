import argparse
from Bio.PDB import *
from os import listdir
from os.path import isfile, join
import numpy as np
import warnings
from Bio import BiopythonWarning
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as cluster
import matplotlib.pyplot as plt
import time


warnings.simplefilter('ignore', BiopythonWarning)

pdb_parser = PDBParser()

amino_acid_dictionary = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
back_bone = ["N", "CA", "C", "O"]

# contact energies dictionary
residue_pairs = {"CYSCYS" : -5.44, "CYSMET" : -4.99, "METCYS" : -4.99, "CYSPHE" : -5.80, "PHECYS" : -5.80, "METMET" : -5.46,
       "METPHE" : -6.56, "PHEMET" : -6.56, "PHEPHE" : -7.26, "CYSILE" : -5.50, "ILECYS" : -5.50, "METILE" : -6.56,
       "ILEMET" : -6.56, "PHEILE" : -6.84, "ILEPHE" : -6.84, "ILEILE" : -6.54, "CYSLEU" : -5.83, "LEUCYS" : -5.83,
       "METLEU" : -6.41, "LEUMET" : -6.41, "PHELEU" : -7.28, "LEUPHE" : -7.28, "ILELEU" : -7.04, "LEUILE" : -7.04,
       "LEULEU" : -7.37, "CYSVAL" : -4.96, "VALCYS" : -4.96, "METVAL" : -5.32, "VALMET" : -5.32, "PHEVAL" : -6.29,
       "VALPHE" : -6.29, "ILEVAL" : -6.05, "VALILE" : -6.05, "LEUVAL" : -6.48, "VALLEU" : -6.48, "VALVAL" : -5.52,
       "CYSTRP" : -4.95, "TRPCYS" : -4.95, "METTRP" : -5.55, "TRPMET" : -5.55, "PHETRP" : -6.16, "TRPPHE" : -6.16,
       "ILETRP" : -5.78, "TRPILE" : -5.78, "LEUTRP" : -6.14, "TRPLEU" : -6.14, "VALTRP" : -5.18, "TRPVAL" : -5.18,
       "TRPTRP" : -5.06, "CYSTYR" : -4.16, "TYRCYS" : -4.16, "METTYR" : -4.91, "TYRMET" : -4.91, "PHETYR" : -5.66,
       "TYRPHE" : -5.66, "ILETYR" : -5.25, "TYRILE" : -5.25, "LEUTYR" : -5.67, "TYRLEU" : -5.67, "VALTYR" : -4.62,
       "TYRVAL" : -4.62, "TRPTYR" : -4.66, "TYRTRP" : -4.66, "TYRTYR" : -4.17, "CYSALA" : -3.57, "ALACYS" : -3.57,
       "METALA" : -3.94, "ALAMET" : -3.94, "PHEALA" : -4.81, "ALAPHE" : -4.81, "ILEALA" : -4.58, "ALAILE" : -4.58,
       "LEUALA" : -4.91, "ALALEU" : -4.91, "VALALA" : -4.04, "ALAVAL" : -4.04, "TRPALA" : -3.82, "ALATRP" : -3.82,
       "TYRALA" : -3.36, "ALATYR" : -3.36, "ALAALA" : -2.72, "CYSGLY" : -3.16, "GLYCYS" : -3.16, "METGLY" : -3.39,
       "GLYMET" : -3.39, "PHEGLY" : -4.13, "GLYPHE" : -4.13, "ILEGLY" : -3.78, "GLYILE" : -3.78, "LEUGLY" : -4.16,
       "GLYLEU" : -4.16, "VALGLY" : -3.38, "GLYVAL" : -3.38, "TRPGLY" : -3.42, "GLYTRP" : -3.42, "TYRGLY" : -3.01,
       "GLYTYR" : -3.01, "ALAGLY" : -2.31, "GLYALA" : -2.31, "GLYGLY" : -2.24, "CYSTHR" : -3.11, "THRCYS" : -3.11,
       "METTHR" : -3.51, "THRMET" : -3.51, "PHETHR" : -4.28, "THRPHE" : -4.28, "ILETHR" : -4.03, "THRILE" : -4.03,
       "LEUTHR" : -4.34, "THRLEU" : -4.34, "VALTHR" : -3.46, "THRVAL" : -3.46, "TRPTHR" : -3.22, "THRTRP" : -3.22,
       "TYRTHR" : -3.01, "THRTYR" : -3.01, "ALATHR" : -2.32, "THRALA" : -2.32, "GLYTHR" : -2.08, "THRGLY" : -2.08,
       "THRTHR" : -2.12, "CYSSER" : -2.86, "SERCYS" : -2.86, "METSER" : -3.03, "SERMET" : -3.03, "PHESER" : -4.02,
       "SERPHE" : -4.02, "ILESER" : -3.52, "SERILE" : -3.52, "LEUSER" : -3.92, "SERLEU" : -3.92, "VALSER" : -3.05,
       "SERVAL" : -3.05, "TRPSER" : -2.99, "SERTRP" : -2.99, "TYRSER" : -2.78, "SERTYR" : -2.78, "ALASER" : -2.01,
       "SERALA" : -2.01, "GLYSER" : -1.82, "SERGLY" : -1.82, "THRSER" : -1.96, "SERTHR" : -1.96, "SERSER" : -1.67,
       "CYSASN" : -2.59, "ASNCYS" : -2.59, "METASN" : -2.95, "ASNMET" : -2.95, "PHEASN" : -3.75, "ASNPHE" : -3.75,
       "ILEASN" : -3.24, "ASNILE" : -3.24, "LEUASN" : -3.74, "ASNLEU" : -3.74, "VALASN" : -2.83, "ASNVAL" : -2.83,
       "TRPASN" : -3.07, "ASNTRP" : -3.07, "TYRASN" : -2.76, "ASNTYR" : -2.76, "ALAASN" : -1.84, "ASNALA" : -1.84,
       "GLYASN" : -1.74, "ASNGLY" : -1.74, "THRASN" : -1.88, "ASNTHR" : -1.88, "SERASN" : -1.58, "ASNSER" : -1.58,
       "ASNASN" : -1.68, "CYSGLN" : -2.85, "GLNCYS" : -2.85, "METGLN" : -3.30, "GLNMET" : -3.30, "PHEGLN" : -4.10,
       "GLNPHE" : -4.10, "ILEGLN" : -3.67, "GLNILE" : -3.67, "LEUGLN" : -4.04, "GLNLEU" : -4.04, "VALGLN" : -3.07,
       "GLNVAL" : -3.07, "TRPGLN" : -3.11, "GLNTRP" : -3.11, "TYRGLN" : -2.97, "GLNTYR" : -2.97, "ALAGLN" : -1.89,
       "GLNALA" : -1.89, "GLYGLN" : -1.66, "GLNGLY" : -1.66, "THRGLN" : -1.90, "GLNTHR" : -1.90, "SERGLN" : -1.49,
       "GLNSER" : -1.49, "ASNGLN" : -1.71, "GLNASN" : -1.71, "GLNGLN" : -1.54, "CYSASP" : -2.41, "ASPCYS" : -2.41,
       "METASP" : -2.57, "ASPMET" : -2.57, "PHEASP" : -3.48, "ASPPHE" : -3.48, "ILEASP" : -3.17, "ASPILE" : -3.17,
       "LEUASP" : -3.40, "ASPLEU" : -3.40, "VALASP" : -2.48, "ASPVAL" : -2.48, "TRPASP" : -2.84, "ASPTRP" : -2.84,
       "TYRASP" : -2.76, "ASPTYR" : -2.76, "ALAASP" : -1.70, "ASPALA" : -1.70, "GLYASP" : -1.59, "ASPGLY" : -1.59,
       "THRASP" : -1.80, "ASPTHR" : -1.80, "SERASP" : -1.63, "ASPSER" : -1.63, "ASNASP" : -1.68, "ASPASN" : -1.68,
       "GLNASP" : -1.46, "ASPGLN" : -1.46, "ASPASP" : -1.21, "CYSGLU" : -2.27, "GLUCYS" : -2.27, "METGLU" : -2.89,
       "GLUMET" : -2.89, "PHEGLU" : -3.56, "GLUPHE" : -3.56, "ILEGLU" : -3.27, "GLUILE" : -3.27, "LEUGLU" : -3.59,
       "GLULEU" : -3.59, "VALGLU" : -2.67, "GLUVAL" : -2.67, "TRPGLU" : -2.99, "GLUTRP" : -2.99, "TYRGLU" : -2.79,
       "GLUTYR" : -2.79, "ALAGLU" : -1.51, "GLUALA" : -1.51, "GLYGLU" : -1.22, "GLUGLY" : -1.22, "THRGLU" : -1.74,
       "GLUTHR" : -1.74, "SERGLU" : -1.48, "GLUSER" : -1.48, "ASNGLU" : -1.51, "GLUASN" : -1.51, "GLNGLU" : -1.42,
       "GLUGLN" : -1.42, "ASPGLU" : -1.02, "GLUASP" : -1.02, "GLUGLU" : -0.91, "CYSHIS" : -3.60, "HISCYS" : -3.60,
       "METHIS" : -3.98, "HISMET" : -3.98, "PHEHIS" : -4.77, "HISPHE" : -4.77, "ILEHIS" : -4.14, "HISILE" : -4.14,
       "LEUHIS" : -4.54, "HISLEU" : -4.54, "VALHIS" : -3.58, "HISVAL" : -3.58, "TRPHIS" : -3.98, "HISTRP" : -3.98,
       "TYRHIS" : -3.52, "HISTYR" : -3.52, "ALAHIS" : -2.41, "HISALA" : -2.41, "GLYHIS" : -2.15, "HISGLY" : -2.15,
       "THRHIS" : -2.42, "HISTHR" : -2.42, "SERHIS" : -2.11, "HISSER" : -2.11, "ASNHIS" : -2.08, "HISASN" : -2.08,
       "GLNHIS" : -1.98, "HISGLN" : -1.98, "ASPHIS" : -2.32, "HISASP" : -2.32, "GLUHIS" : -2.15, "HISGLU" : -2.15,
       "HISHIS" : -3.05, "ARGCYS" : -2.57, "CYSARG" : -2.57, "METARG" : -3.12, "ARGMET" : -3.12, "PHEARG" : -3.98,
       "ARGPHE" : -3.98, "ILEARG" : -3.63, "ARGILE" : -3.63, "LEUARG" : -4.03, "ARGLEU" : -4.03, "VALARG" : -3.07,
       "ARGVAL" : -3.07, "TRPARG" : -3.41, "ARGTRP" : -3.41, "TYRARG" : -3.16, "ARGTYR" : -3.16, "ALAARG" : -1.83,
       "ARGALA" : -1.83, "GLYARG" : -1.72, "ARGGLY" : -1.72, "THRARG" : -1.90, "ARGTHR" : -1.90, "SERARG" : -1.62,
       "ARGSER" : -1.62, "ASNARG" : -1.64, "ARGASN" : -1.64, "GLNARG" : -1.80, "ARGGLN" : -1.80, "ASPARG" : -2.29,
       "ARGASP" : -2.29, "GLUARG" : -2.27, "ARGGLU" : -2.27, "HISARG" : -2.16, "ARGHIS" : -2.16, "ARGARG" : -1.55,
       "CYSLYS" : -1.95, "LYSCYS" : -1.95, "METLYS" : -2.48, "LYSMET" : -2.48, "PHELYS" : -3.36, "LYSPHE" : -3.36,
       "ILELYS" : -3.01, "LYSILE" : -3.01, "LEULYS" : -3.37, "LYSLEU" : -3.37, "VALLYS" : -2.49, "LYSVAL" : -2.49,
       "TRPLYS" : -2.69, "LYSTRP" : -2.69, "TYRLYS" : -2.60, "LYSTYR" : -2.60, "ALALYS" : -1.31, "LYSALA" : -1.31,
       "GLYLYS" : -1.15, "LYSGLY" : -1.15, "THRLYS" : -1.31, "LYSTHR" : -1.31, "SERLYS" : -1.05, "LYSSER" : -1.05,
       "ASNLYS" : -1.21, "LYSASN" : -1.21, "GLNLYS" : -1.29, "LYSGLN" : -1.29, "ASPLYS" : -1.68, "LYSASP" : -1.68,
       "GLULYS" : -1.80, "LYSGLU" : -1.80, "HISLYS" : -1.35, "LYSHIS" : -1.35, "ARGLYS" : -0.59, "LYSARG" : -0.59,
       "LYSLYS" : -0.12, "CYSPRO" : -3.07, "PROCYS" : -3.07, "METPRO" : -3.45, "PROMET" : -3.45, "PHEPRO" : -4.25,
       "PROPHE" : -4.25, "ILEPRO" : -3.76, "PROILE" : -3.76, "LEUPRO" : -4.20, "PROLEU" : -4.20, "VALPRO" : -3.32,
       "PROVAL" : -3.32, "TRPPRO" : -3.73, "PROTRP" : -3.73, "TYRPRO" : -3.19, "PROTYR" : -3.19, "ALAPRO" : -2.03,
       "PROALA" : -2.03, "GLYPRO" : -1.87, "PROGLY" : -1.87, "THRPRO" : -1.90, "PROTHR" : -1.90, "SERPRO" : -1.57,
       "PROSER" : -1.57, "ASNPRO" : -1.53, "PROASN" : -1.53, "GLNPRO" : -1.73, "PROGLN" : -1.73, "ASPPRO" : -1.33,
       "PROASP" : -1.33, "GLUPRO" : -1.26, "PROGLU" : -1.26, "HISPRO" : -2.25, "PROHIS" : -2.25, "ARGPRO" : -1.70,
       "PROARG" : -1.70, "LYSPRO" : -0.97, "PROLYS" : -0.97, "PROPRO" : -1.75}


# Written by Jonas Kuebler & Dominic Boceck & in Python 3.5.2
def get_files(file_directory):
    # creates list of all files in folder
    onlyfiles = [f for f in listdir(file_directory) if isfile(join(file_directory, f))]

    return onlyfiles


def read_structure(structure_name, file):

    structure = pdb_parser.get_structure(structure_name, file)

    return structure


# filters out water and ligands
def filter_primary_structure(structure):

    primary_structure = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in amino_acid_dictionary:
                    primary_structure.append(residue)

    return primary_structure


# get all side chain of every residue
def get_side_chain(primary_structure):

    all_side_chains = []
    side_chain = []

    for residue in primary_structure:
        for atom in residue:
            if atom.get_name() in back_bone:
                continue
            else:
                side_chain.append(atom)
        all_side_chains.append(side_chain)
        side_chain = []

    return all_side_chains


# calculate geometric center
def get_geometric_center(all_side_chains):

    geometric_center = [0, 0, 0]
    centers = []
    atom_count = 0

    for side_chain in all_side_chains:
        for atoms in side_chain:
            atom_count += 1
            geometric_center = [x + y for (x, y) in zip(geometric_center, atoms.get_vector())]
        if atom_count == 0:
            centers.append([])
            atom_count = 0
        else:
            center = [x / atom_count for x in geometric_center]
            centers.append(center)
            geometric_center = [0,0,0]
            atom_count = 0

    return centers


# gets geometric center for glycine
def get_geometric_center_glycine(glycines):

    glycine_centers = []

    for residue in glycines:

        n = residue['N'].get_vector()
        c = residue['C'].get_vector()
        ca = residue['CA'].get_vector()
        # Center at origin
        n = n - ca
        c = c - ca
        # Find rotation matrix that rotates n -120 degrees along the ca-c vector
        rot = rotaxis(-np.pi * 120 / 180.0, c)
        # Apply rotation to ca-n vector
        cb_at_origin = n.left_multiply(rot)
        # Adjust length of cb vector to the given 1.53 Angstrom
        cb_at_origin = Vector(cb_at_origin.normalized()[0] * 1.53, cb_at_origin.normalized()[1] * 1.53,
                              cb_at_origin.normalized()[2] * 1.53)
        # Put on top of ca atom
        cb = cb_at_origin + ca
        # Center of glycine is represented by the calculated cb
        center = cb
        center_array = []

        # convert to array
        for coordinate in center:
            center_array.append(coordinate)
        glycine_centers.append(center_array)

    return glycine_centers


# takes the geometric centers of glycine and puts them into the list of the other centers
def put_centers_together(centers, glycine_centers, indizes):

    counter = 0

    for index in indizes:
        centers[index] = glycine_centers[counter]
        counter += 1

    return centers


# determines if two residues are contact pair
def contact_pairs(all_centers, primary):

    counter_one = 0
    counter_two = 0
    tuple_list = []
    residue_tuples_including_pepbonds = []
    distances_including_pepbonds = []
    residue_tuples_no_pepbonds = []
    distances_no_pepbonds = []

    for center in all_centers:
        for compare_center in all_centers:
            if center != compare_center:
                tmp_center = np.asarray(center)
                tmp_compare_center = np.asarray(compare_center)
                dist = np.linalg.norm(tmp_center - tmp_compare_center)
                if sigmoid_function(dist) >= 0.5:
                    tmp_tuple = [counter_one, counter_two]
                    if not [counter_two, counter_one] in tuple_list:
                        tuple_list.append(tmp_tuple)
                        residue_tuples_including_pepbonds.append([primary[counter_one].get_resname(), primary[counter_two].get_resname()])
                        distances_including_pepbonds.append(sigmoid_function(dist))
                        if not abs(counter_two - counter_one) == 1:
                            residue_tuples_no_pepbonds.append([primary[counter_one].get_resname(), primary[counter_two].get_resname()])
                            distances_no_pepbonds.append(sigmoid_function(dist))

            counter_two += 1
        counter_two = 0
        counter_one += 1

    return [residue_tuples_including_pepbonds, distances_including_pepbonds, residue_tuples_no_pepbonds, distances_no_pepbonds]


def energy_calculation(residue_tuples, distances):

    energy = 0

    index = 0
    for contact_pair in residue_tuples:
        energy = energy + residue_pairs[contact_pair[0] + contact_pair[1]] * distances[index]
        index += 1

    return energy


def get_calpha_atoms(primary_native, primary_conformer):

    calphas_native = []
    calphas_conformer = []
    ids_primary_conformer = []
    ids_primary_native = []
    calphas = []

    for residue in primary_conformer:
        ids_primary_conformer.append(residue.get_id())

    for residue in primary_native:
        ids_primary_native.append(residue.get_id())

    # get lists of c-alpha atoms that occur only in both structures.
    for residue in primary_native:
        if residue.get_id() in ids_primary_conformer:
            calphas_native.append(residue['CA'])


    # get lists of c-alpha atoms that occur only in both structures.
    for residue in primary_conformer:
        if residue.get_id() in ids_primary_native:
            calphas_conformer.append(residue['CA'])

    calphas.append(calphas_native)
    calphas.append(calphas_conformer)

    return calphas


def superimpose(calpha_atoms):

    sup = Superimposer()

    # Specify the atom lists
    # 'fixed' and 'moving' are lists of Atom objects
    # The moving atoms will be put on the fixed atoms
    sup.set_atoms(calpha_atoms[0], calpha_atoms[1])

    # Apply rotation/translation to the moving atoms
    sup.apply(calpha_atoms[1])

    # return rmsd
    return sup.rms


def sigmoid_function(distance):
    result = 1/(1 + np.exp(-1 * (6.5 - distance)))

    return result


def main():
    parser = argparse.ArgumentParser(description="PDB Pipeline")
    parser.add_argument("pdb_directory", help="Directory for your set of conformations")
    parser.add_argument("cry", help="Crystal structure in pdb format")
    args = parser.parse_args()
    crystal = args.cry
    cry_structure = pdb_parser.get_structure('crystal', crystal)

    files = get_files(args.pdb_directory)
    energies = []
    structure_counter = 1
    all_primary_structures = []
    superimpose_rmsd = []

    for file in files:

        # parse structure
        structure = read_structure("Structure: " + str(structure_counter), args.pdb_directory + "/" + file)
        # get primary structrue
        primary_conformer = filter_primary_structure(structure)

        # get calpha atoms
        primary_crystal = filter_primary_structure(cry_structure)
        calphas = get_calpha_atoms(primary_crystal, primary_conformer)

        # calculate superimposed RMSD of every file and append it to list
        # with filename so we know which conformation the rmsd belongs to
        superimpose_rmsd.append([superimpose(calphas), file])

        all_primary_structures.append(primary_conformer)
        index = 0
        # get index of GLY in primary structure
        glycines = []
        indizes = []
        for residue in primary_conformer:
            if residue.get_resname() == "GLY":
                glycines.append(residue)
                indizes.append(index)
            index += 1

        # get side chains
        side_chains = get_side_chain(primary_conformer)

        # get geomertric centers and put them together
        centers = get_geometric_center(side_chains)
        glycine_centers = get_geometric_center_glycine(glycines)
        all_centers = put_centers_together(centers, glycine_centers, indizes)

        residue_tuples = contact_pairs(all_centers, primary_conformer)

        energy_with_pepbonds = energy_calculation(residue_tuples[0], residue_tuples[1])
        energy_no_pepbonds = energy_calculation(residue_tuples[2], residue_tuples[3])

        energies.append([file, energy_with_pepbonds, energy_no_pepbonds])

        structure_counter += 1


        print(energies)
    # sort rsmd of superimpose to get the best and worst
    superimpose_rmsd = sorted(superimpose_rmsd)
    print(superimpose_rmsd)


if __name__ == "__main__":
    main()


