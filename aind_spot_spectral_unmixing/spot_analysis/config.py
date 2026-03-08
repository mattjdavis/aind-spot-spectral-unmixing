import pathlib
from typing import Dict, Any, List, Set
import os
import re
import json


class Config():
    # take dataset_folder as init
    

    

    # Gene dictionary
    DEFAULT_GENE_DICT: Dict[str, Dict[str, str]] = {'0':{'1': 'Vip', '2': 'Sst', '4': 'Slc17a7'},
            '1':{'1': 'Cbln4', '2': 'Cdk18', '3': 'Kcnab1', '4': 'Nos1'},
            '2':{'1': 'Adcyap1', '2': 'Rorb', '3': 'Myh7', '4': 'Pdyn'},
            '3':{'1': 'Wfs1', '2': 'Npnt', '3': 'F2r12', '4': 'Trp53i11'},
            '4':{'1': 'Thsd7a', '2': 'Syt6', '3': 'Car4', '4': 'Tmem215'},
            '5':{'1': 'Pvalb', '2': 'Olig1', '3': 'Lypd1', '4': 'Synpr'},
            '6':{'1': 'Parm1', '2': 'Sfrp2', '3': 'Tnnc1', '4': 'Penk'},
            '7':{'1': 'Etv1', '2': 'Lsp1', '3': 'Slc18a3', '4': 'Calb1'},
            '8':{'1': 'Alcam', '2': 'Cidea', '3': 'Prss23', '4': 'Il1rap12'},
            '9':{'1': 'Cplx', '2': 'Ctss', '3': 'Npy',},
            '10':{'1': 'Slc18a8', '2': 'Tshz2', '3': 'Egln3', '4': 'Lpl'},
            '11':{'1': 'Gad2', '2': 'Ostn', '3': 'Lhx6', '4': 'Stk17b'},
            '12':{'1': 'Cck', '2': 'Crispld2', '3': 'Nmbr', '4': 'Anxa2'},
            '13':{'1': 'Snap25', '2': 'lgfbp4', '3': 'Chrm2', '4': 'Ndnf'}}

    def __init__(self, dataset_folder: str, manifest: Dict[str, Any] = None):
        self.dataset_folder = pathlib.Path(dataset_folder)

        self.SPOTS_FOLDER = pathlib.Path(f'/data/{self.dataset_folder}')
        self.DATA_FOLDER = pathlib.Path(f'/data/{self.dataset_folder}')

        # set spots for class
        Config.SPOTS_FOLDER = self.SPOTS_FOLDER
        Config.DATA_FOLDER = self.DATA_FOLDER

        if manifest is not None:
            # Use the provided in-memory manifest (e.g. with round/gene-name overrides)
            # instead of re-reading from disk.
            self.manifest = manifest
        else:
            self._load_manifest()
        self._update_round_from_manifest()
        self._make_gene_dict_from_manifest()
        self.folder_paths = None
        self.folder_paths = self.get_and_validate_folder_paths()

        

    OUTPUT_FOLDER = pathlib.Path(f'/results/')
    OUTPUT_DATA_TYPE = 'zarr'
    SCRATCH_FOLDER = pathlib.Path('/scratch/')
    
    # Processing parameters
    _default_ROUND_N = 0
    MIN_DISTS = 5
    PERCENTILE = 95
    
    # Demixing parameters
    FRAC_SAMPLED = 0.1
    N_SUBSET = 100000
    EPOCHS = 10000
    RESAMPLE_ITER = 50
    L1 = 0
    LEARNING_RATE = 1e-9
    
    # QC parameters --- these are getting moved to qc capsule
    CENT_CUTOFF = 1
    CORR_CUTOFF = 0.5
    DIST_CUTOFF = 4
    # cell by gene table parameters    
    min_dist = 3
    volume_quantiles = (0.08, 0.5, 0.95)


    folder_paths = None

        

    #@classmethod
    #def _load_manifest(cls):
    #    """Load the processing manifest JSON file"""

        # manifest_path = pathlib.Path(cls.dataset_name) / 'derived' / 'processing_manifest.json'
    #    manifest_path = list(pathlib.Path(cls.DATA_FOLDER).glob("derived/processing_manifest.json"))
        
    
    #    if not len(manifest_path):
    #        print(f'didnt find pipeline processing manifest')
            #raise FileNotFoundError("No processing_manifest.json was found!")
        
    #        manifest_path = list(pathlib.Path(cls.DATA_FOLDER).glob("*/derived/processing_manifest.json"))
    #        if not len(manifest_path):
    #            raise FileNotFoundError("No capsule processing_manifest.json was found!")

        
    #    print(f'Manifest_path {manifest_path}')

    #    try:
    #        with open(manifest_path[0], 'r') as f:
    #            cls.manifest = json.load(f)
    #    except FileNotFoundError:
    #        cls.manifest = None
    #        raise FileNotFoundError(f"Processing manifest not found at {manifest_path}")

    # set SPOTS_FOLDER classmethod
    def _set_spots_folder(self):
        self.SPOTS_FOLDER = pathlib.Path(f'/data/{self.dataset_folder}')

    def _set_data_folder(self):
        self.DATA_FOLDER = pathlib.Path(f'/data/{self.dataset_folder}')

    def _update_round_from_manifest(self):
        if not self.manifest:
            self.ROUND_N = self._default_ROUND_N
            return
        round = self.manifest['round']
        # if round != -1: 
        self.ROUND_N = round
        # else:
            # self.ROUND_N = self._default_ROUND_N

        """ Processing Manifest Json Example
    {'segmentation_channels': {'background': '405', 'nuclear': None}, 'spot_channels': ['561', '488', '638'], 'round': 1, 'stitching_channels': ['561', '488', '638'], 'gene_dict': {'405': {'gene': 'Rn28s', 'barcode': '', 'fluorophore': '', 'wavelength': 'dtype:', 'round': 1}, '561': {'gene': 'Calb2', 'barcode': 'B7', 'fluorophore': '', 'wavelength': '561,', 'round': 1}, '488': {'gene': 'Npy', 'barcode': 'B1', 'fluorophore': '', 'wavelength': '488,', 'round': 1}, '638': {'gene': 'Tac1', 'barcode': 'B3', 'fluorophore': '', 'wavelength': '638,', 'round': 1}}}"""

    @staticmethod
    def _str_channel(ch) -> str:
        """Coerce a channel value to str, stripping any accidental whitespace."""
        return str(ch).strip()

    def _make_gene_dict_from_manifest(self):
        """Make a gene_dict from the processing manifest"""
        if not self.manifest:
            self.GENE_DICT = self.DEFAULT_GENE_DICT
            return
        spot_channels = self.manifest['spot_channels']
        round = self.manifest['round']
        manifest_gene_dict = self.manifest['gene_dict']        #gene_dict is a dict of dicts with keys: round { channel: gene_name}
        temp_dict = {}
        
        for channel, gene in manifest_gene_dict.items():
            temp_dict[self._str_channel(channel)] = str(gene['gene'])
        
        gene_dict= {}
        gene_dict[str(round)] = temp_dict
        self.GENE_DICT = gene_dict


    def get_round_channels(self) -> Dict[str, str]:
        return self.GENE_DICT[str(self.ROUND_N)]


    def get_round_spot_channels(self) -> Dict[str, str]:
        spot_channels = self.manifest['spot_channels']
        return [self._str_channel(ch) for ch in spot_channels]

    def get_folder_paths(self) -> Dict[str, Dict[str, str]]:
        return self.get_and_validate_folder_paths()

    
    
    def _load_manifest(self):
        """Load the processing manifest JSON file"""
        # First try: <DATA_FOLDER>/derived/processing_manifest.json  (pipeline path)
        manifest_path = list(pathlib.Path(self.DATA_FOLDER).glob("derived/processing_manifest.json"))

        if not len(manifest_path):
            print('Didn\'t find manifest in derived/, trying */derived/...')
            manifest_path = list(pathlib.Path(self.DATA_FOLDER).glob("*/derived/processing_manifest.json"))

        if not len(manifest_path):
            # Second try: <DATA_FOLDER>/processing_manifest.json  (root-level, matches hcr_dataset behaviour)
            print('Didn\'t find manifest in */derived/, trying root-level...')
            manifest_path = list(pathlib.Path(self.DATA_FOLDER).glob("processing_manifest.json"))

        if not len(manifest_path):
            raise FileNotFoundError("No capsule processing_manifest.json was found!")

        print(f'Manifest_path {manifest_path}')

        try:
            with open(manifest_path[0], 'r') as f:
                self.manifest = json.load(f)
                print(f"Loaded manifest with channels: {self.manifest.get('spot_channels', [])}")

                # Update round to 5 for specific datasets
                dataset_folder_str = str(self.dataset_folder.name) if isinstance(self.dataset_folder, pathlib.Path) else str(self.dataset_folder)
                if dataset_folder_str in ("HCR_754803_2025-09-18_13-00-00_processed_2025-09-20_22-57-09",
                                        "HCR_767018_2025-09-18_13-00-00_processed_2025-09-20_22-57-09"):
                    print(f"Updating round from {self.manifest.get('round')} to 5 for dataset {dataset_folder_str}")
                    self.manifest["round"] = 5
        except FileNotFoundError:
            self.manifest = None
            raise FileNotFoundError(f"Processing manifest not found at {manifest_path}")

    def get_folder_paths_pipeline(self) -> Dict[str, Dict[str, str]]: #get_folder_paths_pipeline
        """Returns folder paths from what is attached in /data/"""
        spot_regex = r".*(\d{1,3})_stats\/image_data_.*_(\d{1,3})_versus_spots_(\d{1,3})\.csv"
        exclude = set(['*.zarr'])
        spots_folders = {}
        multichan_folders = {}

        for root, dirs, files in os.walk(self.DATA_FOLDER):
            # Exclude .zarr directories
            dirs[:] = [d for d in dirs if not d.endswith('.zarr')]
            for file in files:
                # Skip files within .zarr directories
                if '.zarr' in root:
                    continue
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, self.DATA_FOLDER)

                # Check for spot intensity files
                spot_match = re.match(spot_regex, relative_path)
                if spot_match:
                    source_channel = self._str_channel(spot_match.group(2))
                    target_channel = self._str_channel(spot_match.group(3))

                    if source_channel == target_channel: 
                        spots_folders[source_channel] = relative_path
                    else:
                        if multichan_folders == {} or source_channel not in multichan_folders.keys():
                            multichan_folders[source_channel]= {target_channel: relative_path}
                        else: 
                            multichan_folders[source_channel][target_channel] = relative_path
        return {
            'spots_folders': spots_folders,
            'multichan_folders': multichan_folders
        }

    def validate_folder_paths(self, folder_paths: Dict[str, Dict[str, str]]) -> None:
        """Validates the generated folder paths"""
        expected_channels = set(self.get_round_channels().keys())

        # Validate spots folders
        spots_channels = set(folder_paths['spots_folders'].keys())
        if spots_channels != expected_channels:
            missing = expected_channels - spots_channels
            extra = spots_channels - expected_channels
            #print(f"Warning: Mismatch in spots folders. Missing: {missing}, Extra: {extra}")

        # Validate multichannel folders
        multichan_channels = set(folder_paths['multichan_folders'].keys())
        if multichan_channels != expected_channels:
            missing = expected_channels - multichan_channels
            extra = multichan_channels - expected_channels
            #print(f"Warning: Mismatch in multichannel folders. Missing: {missing}, Extra: {extra}")

        for source_channel, targets in folder_paths['multichan_folders'].items():
            expected_targets = expected_channels - {source_channel}
            if set(targets.keys()) != expected_targets:
                missing = expected_targets - set(targets.keys())
                extra = set(targets.keys()) - expected_targets
                #print(f"Warning: Mismatch in multichannel targets for channel {source_channel}. Missing: {missing}, Extra: {extra}")

    def get_and_validate_folder_paths(self) -> Dict[str, Dict[str, str]]:
        """Gets folder paths and validates them"""
        if self.folder_paths == None: 
            folder_paths = self.get_folder_paths_pipeline()
            #print(f'folder_paths {folder_paths}')

            self.validate_folder_paths(folder_paths)
            self.folder_paths = folder_paths
        else: 
            return self.folder_paths
