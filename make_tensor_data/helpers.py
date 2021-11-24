import re
import numpy as np
from collections import OrderedDict

import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.io import read, write

_mprops = {
    'ms': ('sigma', 1),
    'sus': ('S', 0),
    'efg': ('V', 1),
    'isc': ('K', 2)}
# (matrix name, number of atoms in interaction) for various magres quantities

def generate_status_dict(PATH, status="FAIL"):
    """Builds a dictionary from an extended xyz file 
    containing key: CSD-NAME value:status pairs
    """
    extxyz = read(PATH,format="extxyz",index=':')
    status_dict = {atom.info["NAME"]: status for atom in extxyz}
    return status_dict

def build_extxy_dict(PATH):
    """Builds a dictionary from an extended xyz file 
    containing key: CSD-NAME value: atoms object pairs
    """
    extxyz = read(PATH,format="extxyz",index=':')
    status_dict = {atom.info["NAME"]: atom for atom in extxyz}
    return status_dict


def test_status(PATH_TENSOR,outlierdict):
    """Test wether status from a dict {CSD-Name:status} was correctly written to
    another extended .xyz file located at PATH_TENSOR
    """
    status_list = []
    structures = read(PATH_TENSOR,format="extxyz",index=":")
    for structure in structures:
        status = structure.info["STATUS"]
        name = structure.info["NAME"]
        status_list.append(status == outlierdict[name])
    print(status_list)    
    return all(status_list)

def compaire(struct_iso,struct_tensor,print_why=False):
    """This function compares two atoms objects.
        struct_iso contains coordinates and an additional atoms.arrays["CS"]
        with the atom wise isotropic shifts
        struct_tensor contains coordinates and two additional atoms.arrays["cs_tensor"]
        and atoms.arrays["cs_shifts"]. Coordinates, Cell_info, PBC and isotropic shifts 
        between both are checked for equivalence upto an absolute tolerance of 1e-08 A
        isotropic shifts of both objects are checked for equivalence
        with an absolute tolerance of 1e-02 ppm
        The atomic shift tensor of the struct_tensor object is diagonalized 
        and the average of the diagonal elements is checked for equivalence against the
        isotropic shifts of struct_iso with an absolute tolerance of 1e-02 ppm 
    """
    
    conditions = []
    conditions.append(np.allclose(struct_iso.get_positions(),struct_tensor.get_positions(),atol=1e-08))
    conditions.append(np.allclose(struct_iso.get_cell(),struct_tensor.get_cell(),atol=1e-08))
    conditions.append(np.allclose(struct_iso.get_pbc(),struct_tensor.get_pbc()))
    
    shift_list = []
    
    for tensor in struct_tensor.arrays["cs_tensor"]:
        iso_shift = np.sum(np.linalg.eig(tensor.reshape((3,3)))[0])/3
        shift_list.append(iso_shift)
    
    conditions.append(np.allclose(np.array(shift_list),struct_iso.arrays["CS"],atol=1e-02))
    
    try:
        shift_stored = struct_tensor.arrays["cs_iso"]
        conditions.append(np.allclose(np.array(shift_list),shift_stored,atol=1e-02))
    except:
        pass
    if print_why is True:
        print(conditions)
    #print(conditions)  
    #print(shift_list)
    #print(struct_iso.arrays["CS"])
    return all(conditions)

def check_plausibility(TENSORPATH,ISOPATH):
    """This function checks if shift tensors and isotropic shifts are correctly written to the 
    extended .xyz file.
    Takes the path of the extended .xyz file with the shift tensors TENSORPATH and the 
    originally reported ISOPATH file as input.
    """
    #build iso entry dicts
    structs_iso = read(ISOPATH,index=":",format="extxyz")
    iso_dict = {atoms.info["NAME"]: atoms for atoms in structs_iso}
    #for struct in structs_iso:
    #    struct.wrap()
    
    #tensor structs_list
    structs_tensor = read(TENSORPATH,index=":",format="extxyz")
    #for struct in structs_tensor:
    #    struct.wrap()
    
    for n, atoms_tens in enumerate(structs_tensor):
        name = atoms_tens.info["NAME"]
        atoms_iso = iso_dict[name]
        passing = compaire(atoms_iso,atoms_tens)
        if passing is True:
            continue
        else:
            print(n)
            print(name)
            compaire(atoms_iso,atoms_tens,print_why=True)
            print(np.sum(np.linalg.norm(atoms_iso.get_positions()-atoms_tens.get_positions(),axis=1)))
            continue
    
    return None




#This is the modified magres reader from:
#https://gitlab.com/ase/ase/-/blob/master/ase/io/magres.py
#TODO: include license ?
def read_magres_modified(fd, include_unrecognised=False):
    """
        Reader function for magres files.
    """

    blocks_re = re.compile(r'[\[<](?P<block_name>.*?)[>\]](.*?)[<\[]/' +
                           r'(?P=block_name)[\]>]', re.M | re.S)

    """
    Here are defined the various functions required to parse
    different blocks.
    """

    def tensor33(x):
        return np.squeeze(np.reshape(x, (3, 3))).tolist()

    def tensor31(x):
        return np.squeeze(np.reshape(x, (3, 1))).tolist()

    def get_version(file_contents):
        """
            Look for and parse the magres file format version line
        """

        lines = file_contents.split('\n')
        match = re.match(r'\#\$magres-abinitio-v([0-9]+).([0-9]+)', lines[0])

        if match:
            version = match.groups()
            version = tuple(vnum for vnum in version)
        else:
            version = None

        return version

    def parse_blocks(file_contents):
        """
            Parse series of XML-like deliminated blocks into a list of
            (block_name, contents) tuples
        """

        blocks = blocks_re.findall(file_contents)

        return blocks

    def parse_block(block):
        """
            Parse block contents into a series of (tag, data) records
        """

        def clean_line(line):
            # Remove comments and whitespace at start and ends of line
            line = re.sub('#(.*?)\n', '', line)
            line = line.strip()

            return line

        name, data = block

        lines = [clean_line(line) for line in data.split('\n')]

        records = []

        for line in lines:
            #line = line.replace("-", " -")
            xs = line.split()

            if len(xs) > 0:
                tag = xs[0]
                data = xs[1:]

                records.append((tag, data))

        return (name, records)

    def check_units(d):
        """
            Verify that given units for a particular tag are correct.
        """

        allowed_units = {'lattice': 'Angstrom',
                         'atom': 'Angstrom',
                         'ms': 'ppm',
                         'efg': 'au',
                         'efg_local': 'au',
                         'efg_nonlocal': 'au',
                         'isc': '10^19.T^2.J^-1',
                         'isc_fc': '10^19.T^2.J^-1',
                         'isc_orbital_p': '10^19.T^2.J^-1',
                         'isc_orbital_d': '10^19.T^2.J^-1',
                         'isc_spin': '10^19.T^2.J^-1',
                         'isc': '10^19.T^2.J^-1',
                         'sus': '10^-6.cm^3.mol^-1',
                         'calc_cutoffenergy': 'Hartree', }

        if d[0] in d and d[1] == allowed_units[d[0]]:
            pass
        else:
            raise RuntimeError('Unrecognized units: %s %s' % (d[0], d[1]))

        return d

    def parse_magres_block(block):
        """
            Parse magres block into data dictionary given list of record
            tuples.
        """

        name, records = block

        # 3x3 tensor
        def ntensor33(name):
            return lambda d: {name: tensor33([float(x) for x in data])}

        # Atom label, atom index and 3x3 tensor
        def sitensor33(name):
            return lambda d: {'atom': {'label': data[0],
                                       'index': int(data[1])},
                              name: tensor33([float(x) for x in data[2:]])}

        # 2x(Atom label, atom index) and 3x3 tensor
        def sisitensor33(name):
            return lambda d: {'atom1': {'label': data[0],
                                        'index': int(data[1])},
                              'atom2': {'label': data[2],
                                        'index': int(data[3])},
                              name: tensor33([float(x) for x in data[4:]])}

        tags = {'ms': sitensor33('sigma'),
                'sus': ntensor33('S'),
                'efg': sitensor33('V'),
                'efg_local': sitensor33('V'),
                'efg_nonlocal': sitensor33('V'),
                'isc': sisitensor33('K'),
                'isc_fc': sisitensor33('K'),
                'isc_spin': sisitensor33('K'),
                'isc_orbital_p': sisitensor33('K'),
                'isc_orbital_d': sisitensor33('K'),
                'units': check_units}

        data_dict = {}

        for record in records:
            tag, data = record

            if tag not in data_dict:
                data_dict[tag] = []

            data_dict[tag].append(tags[tag](data))

        return data_dict

    def parse_atoms_block(block):
        """
            Parse atoms block into data dictionary given list of record tuples.
        """

        name, records = block

        # Lattice record: a1, a2 a3, b1, b2, b3, c1, c2 c3
        def lattice(d):
            return tensor33([float(x) for x in data])

        # Atom record: label, index, x, y, z
        def atom(d):
            return {'species': data[0],
                    'label': data[1],
                    'index': int(data[2]),
                    'position': tensor31([float(x) for x in data[3:]])}

        def symmetry(d):
            return ' '.join(data)

        tags = {'lattice': lattice,
                'atom': atom,
                'units': check_units,
                'symmetry': symmetry}

        data_dict = {}

        for record in records:
            tag, data = record
            if tag not in data_dict:
                data_dict[tag] = []
            data_dict[tag].append(tags[tag](data))

        return data_dict

    def parse_generic_block(block):
        """
            Parse any other block into data dictionary given list of record
            tuples.
        """

        name, records = block

        data_dict = {}

        for record in records:
            tag, data = record

            if tag not in data_dict:
                data_dict[tag] = []

            data_dict[tag].append(data)

        return data_dict

    """
        Actual parser code.
    """

    block_parsers = {'magres': parse_magres_block,
                     'atoms': parse_atoms_block,
                     'calculation': parse_generic_block, }

    file_contents = fd

    # This works as a validity check
    version = get_version(file_contents)
    if version is None:
        # This isn't even a .magres file!
        raise RuntimeError('File is not in standard Magres format')
    blocks = parse_blocks(file_contents)

    data_dict = {}

    for block_data in blocks:
        block = parse_block(block_data)

        if block[0] in block_parsers:
            block_dict = block_parsers[block[0]](block)
            data_dict[block[0]] = block_dict
        else:
            # Throw in the text content of blocks we don't recognise
            if include_unrecognised:
                data_dict[block[0]] = block_data[1]

    # Now the loaded data must be turned into an ASE Atoms object

    # First check if the file is even viable
    if 'atoms' not in data_dict:
        raise RuntimeError('Magres file does not contain structure data')

    # Allowed units handling. This is redundant for now but
    # could turn out useful in the future

    magres_units = {'Angstrom': ase.units.Ang}

    # Lattice parameters?
    if 'lattice' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['lattice']
        except KeyError:
            raise RuntimeError('No units detected in file for lattice')
        u = magres_units[u]
        cell = np.array(data_dict['atoms']['lattice'][0]) * u
        pbc = True
    else:
        cell = None
        pbc = False

    # Now the atoms
    symbols = []
    positions = []
    indices = []
    labels = []

    if 'atom' in data_dict['atoms']:
        try:
            u = dict(data_dict['atoms']['units'])['atom']
        except KeyError:
            raise RuntimeError('No units detected in file for atom positions')
        u = magres_units[u]
        # Now we have to account for the possibility of there being CASTEP
        # 'custom' species amongst the symbols
        custom_species = None
        for a in data_dict['atoms']['atom']:
            spec_custom = a['species'].split(':', 1)
            if len(spec_custom) > 1 and custom_species is None:
                # Add it to the custom info!
                custom_species = list(symbols)
            symbols.append(spec_custom[0])
            positions.append(a['position'])
            indices.append(a['index'])
            labels.append(a['label'])
            if custom_species is not None:
                custom_species.append(a['species'])

    atoms = Atoms(cell=cell,
                  pbc=pbc,
                  symbols=symbols,
                  positions=positions)

    # Add custom species if present
    if custom_species is not None:
        atoms.new_array('castep_custom_species', np.array(custom_species))

    # Add the spacegroup, if present and recognizable
    if 'symmetry' in data_dict['atoms']:
        try:
            spg = Spacegroup(data_dict['atoms']['symmetry'][0])
        except SpacegroupNotFoundError:
            # Not found
            spg = Spacegroup(1)  # Most generic one
        atoms.info['spacegroup'] = spg

    # Set up the rest of the properties as arrays
    atoms.new_array('indices', np.array(indices))
    atoms.new_array('labels', np.array(labels))

    # Now for the magres specific stuff
    li_list = list(zip(labels, indices))

    def create_magres_array(name, order, block):

        if order == 1:
            u_arr = [None] * len(li_list)
        elif order == 2:
            u_arr = [[None] * (i + 1) for i in range(len(li_list))]
        else:
            raise ValueError(
                'Invalid order value passed to create_magres_array')

        for s in block:
            # Find the atom index/indices
            if order == 1:
                # First find out which atom this is
                at = (s['atom']['label'], s['atom']['index'])
                try:
                    ai = li_list.index(at)
                except ValueError:
                    raise RuntimeError('Invalid data in magres block')
                # Then add the relevant quantity
                u_arr[ai] = s[mn]
            else:
                at1 = (s['atom1']['label'], s['atom1']['index'])
                at2 = (s['atom2']['label'], s['atom2']['index'])
                ai1 = li_list.index(at1)
                ai2 = li_list.index(at2)
                # Sort them
                ai1, ai2 = sorted((ai1, ai2), reverse=True)
                u_arr[ai1][ai2] = s[mn]

        if order == 1:
            return np.array(u_arr)
        else:
            return np.array(u_arr, dtype=object)

    if 'magres' in data_dict:
        if 'units' in data_dict['magres']:
            atoms.info['magres_units'] = dict(data_dict['magres']['units'])
            for u in atoms.info['magres_units']:
                # This bit to keep track of tags
                u0 = u.split('_')[0]

                if u0 not in _mprops:
                    raise RuntimeError('Invalid data in magres block')

                mn, order = _mprops[u0]

                if order > 0:
                    u_arr = create_magres_array(mn, order,
                                                data_dict['magres'][u])
                    atoms.new_array(u, u_arr)
                else:
                    # atoms.info['magres_data'] = atoms.info.get('magres_data',
                    #                                            {})
                    # # We only take element 0 because for this sort of data
                    # # there should be only that
                    # atoms.info['magres_data'][u] = \
                    #     data_dict['magres'][u][0][mn]
                    if atoms.calc is None:
                        calc = SinglePointDFTCalculator(atoms)
                        atoms.calc = calc
                        atoms.calc.results[u] = data_dict['magres'][u][0][mn]

    if 'calculation' in data_dict:
        atoms.info['magresblock_calculation'] = data_dict['calculation']

    if include_unrecognised:
        for b in data_dict:
            if b not in block_parsers:
                atoms.info['magresblock_' + b] = data_dict[b]

    return atoms



def tensor_string(tensor):
    return ' '.join(' '.join(str(x) for x in xs) for xs in tensor)