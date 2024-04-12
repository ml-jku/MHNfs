import pytorch_lightning as pl
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

class FSMolDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """
        Init for lightning data module
        :param config: hydra config.yaml
        """
        super().__init__()
        self.config = config

    def setup(self, stage=None) -> None:
        """
        Prepares data for every GPU which is used for training
        :param stage: None
        """

        #  Load training validation and test data
        self.databaseTraining = self._load_preprocessed_data(fold="training")
        self.databaseValidation = self._load_preprocessed_data(fold="validation")
        self.databaseTest = self._load_preprocessed_data(fold="test")

        # Draw support sets and clean query sets for validation and test data
        self.databaseValidation = self._draw_fixed_support_and_query_set(
            self.databaseValidation
        )
        self.databaseTest = self._draw_fixed_support_and_query_set(self.databaseTest)

        # Training data class (including .__getitem__(index))
        self.trainingData = self._TrainingData(self.databaseTraining, self.config)

        # Validation data class (including .__getitem__(index))
        self.validationData = self._EvalData(self.databaseValidation, self.config)

        # Test data
        self.testData = self._EvalData(self.databaseTest, self.config)

    def train_dataloader(self):
        """
        Dataloader for training
        :return: Batches of the training data
        """
        return DataLoader(
            self.trainingData,
            batch_size=self.config.model.training.batch_size,
            shuffle=True,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    def val_dataloader(self):
        """
        Dataloader for validation data
        :return: Batches of the validation data
        """

        return DataLoader(
            self.validationData,
            batch_size=self.config.validation.batch_size,
            shuffle=False,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    def test_dataloader(self):
        """
        Dataloader for test data
        :return: Batches of the test data
        """

        return DataLoader(
            self.testData,
            batch_size=self.config.test.batch_size,
            num_workers=self.config.system.ressources.num_workers_cpu,
        )

    # Custom functions and classes
    def _load_preprocessed_data(self, fold=["training", "validation", "test"]):
        """
        This function loads the preprocessed training, validation or test data
        :param fold: training, validation, test
        :return: dictionary which stores the data
        """
        if fold == "training":
            path = self.config.system.data.path + self.config.system.data.dir_training
        elif fold == "validation":
            path = self.config.system.data.path + self.config.system.data.dir_validation
        elif fold == "test":
            path = self.config.system.data.path + self.config.system.data.dir_test

        # "Data triplet": (molecule index, task index, label)
        molIds = np.load(
            path + self.config.system.data.name_mol_ids
        )  # molecule indices for triplets
        taskIds = np.load(
            path + self.config.system.data.name_target_ids
        )  # target indices for triplets
        labels = np.load(path + self.config.system.data.name_labels).astype(
            "float32"
        )  # labels for triplets
        molInputs = np.load(path + self.config.system.data.name_mol_inputs).astype(
            "float32"
        )  # molecule data base (fingerprints, descriptors)
        dictMolSmilesid = pickle.load(
            open(path + self.config.system.data.name_dict_mol_smiles_id, "rb")
        )  # connects molecule index wuth original SMILES
        dictTaskidActivemolecules = pickle.load(
            open(
                path + self.config.system.data.name_dict_target_id_activeMolecules, "rb"
            )
        )  # stores molecule indices of active mols for each target
        dictTaskidInactivemolecules = pickle.load(
            open(
                path + self.config.system.data.name_dict_target_id_inactiveMolecules,
                "rb",
            )
        )  # stores molecule indices of inactive mols for each target
        dictTasknamesId = pickle.load(
            open(path + self.config.system.data.name_dict_target_names_id, "rb")
        )  # connects target indices with original target names

        dataDict = {
            "molIds": molIds,
            "taskIds": taskIds,
            "labels": labels,
            "molInputs": molInputs,
            "dictMolSmilesid": dictMolSmilesid,
            "dictTaskidActivemolecules": dictTaskidActivemolecules,
            "dictTaskidInactivemolecules": dictTaskidInactivemolecules,
            "dictTasknamesId": dictTasknamesId,
        }
        return dataDict

    def _draw_fixed_support_and_query_set(self, dataDict):
        """
        This functions draws a fixed support set for each task. The query set consists of all molecules which are not
        put into the support set.

        Support and query set splitting can be done by:
        - Providing specific numbers for active and inactive molecules in the support set
          See config.supportSet.count
        - Providing splitting ratios for actives and inactives
          See config-supportSet.ratio
        - Performing a stratified random split
          See config.supportSet.stratified
        The config controls the type of splitting.

        :param dataDict: dictionary in which the data for validation or test is stored
        :return: Support set, including indices for active and inactive molecules. Query set
        """

        # prepare keys for query Set
        dataDict["query_molIds"] = []
        dataDict["query_taskIds"] = []
        dataDict["query_labels"] = []
        dataDict["supportSetActives"] = {}
        dataDict["supportSetInactives"] = {}

        # Define splitting functions
        def stratified_splitting_eval(
            dataDict, active_mols_in_task, inactive_mols_in_task, task_idx
        ):
            """
            This functions performs a stratified shuffled random split to create the support and the query set.
            Since this function lives within the _draw_fixed_support_and_query_set function the drawn support sets are
            fixed and the dataDict is updated.
            :param dataDict: data base created by _load_preprocessed_data
            :param active_mols_in_task: list of indices
            :param inactive_mols_in_task: list of indices
            :param task_idx: task index
            """

            # Create labels for molecules
            activeSet_labels = np.ones(len(active_mols_in_task)).reshape(-1, 1)
            inactiveSet_labels = np.zeros(len(inactive_mols_in_task)).reshape(-1, 1)

            # Merge active and inactive molecules
            activeAndInactiveSet = np.array(active_mols_in_task + inactive_mols_in_task)
            activeAndInactiveSet_labels = np.vstack(
                [activeSet_labels, inactiveSet_labels]
            ).reshape(-1)

            # Create splitter object
            skf = StratifiedShuffleSplit(
                n_splits=1, test_size=self.config.supportSet.supportSetSize
            )

            # Split data into query and support set
            querySetSplitIndices, supportSetSplitIndices = list(
                skf.split(activeAndInactiveSet, activeAndInactiveSet_labels)
            )[0]

            # Prepare support set
            suppportSetActiveAndInactive_ids = activeAndInactiveSet[
                supportSetSplitIndices
            ]
            suppportSetActiveAndInactive_labels = activeAndInactiveSet_labels[
                supportSetSplitIndices
            ]

            supportSetActivesIds = suppportSetActiveAndInactive_ids[
                suppportSetActiveAndInactive_labels == 1
            ]
            supportSetInactivesIds = suppportSetActiveAndInactive_ids[
                suppportSetActiveAndInactive_labels == 0
            ]

            # Prepare query set
            querySetActiveAndInactive_ids = activeAndInactiveSet[querySetSplitIndices]
            querySetActiveAndInactive_labels = activeAndInactiveSet_labels[
                querySetSplitIndices
            ]

            # Include everything into datadict
            dataDict["query_molIds"] += list(querySetActiveAndInactive_ids)
            dataDict["query_taskIds"] += list(
                np.repeat(task_idx, len(querySetActiveAndInactive_ids))
            )
            dataDict["query_labels"] += list(querySetActiveAndInactive_labels)
            dataDict["supportSetActives"][task_idx] = list(supportSetActivesIds)
            dataDict["supportSetInactives"][task_idx] = list(supportSetInactivesIds)

            return dataDict

            # Loop over tasks

        for task_idx in list(dataDict["dictTaskidActivemolecules"]):
            # Collect indices of active and inactive molecules in task
            active_mols_in_task = dataDict["dictTaskidActivemolecules"][task_idx]
            inactive_mols_in_task = dataDict["dictTaskidInactivemolecules"][task_idx]
            
            dataDict = stratified_splitting_eval(
                dataDict, active_mols_in_task, inactive_mols_in_task, task_idx)

        return dataDict

    class _TrainingData:
        """
        This is the pytorch dataclass which is required by the training dataloader.
        - __getitem__ returns a data point in triplet format ( [mol_id, task_id, label] ) and the referring support
          set
        - __len__ returns the number of training data points
        """

        def __init__(self, database, config):
            self.database = database
            self.config = config

            self.len = len(self.database["molIds"])

        def __getitem__(self, index):
            # Get triplet
            molIdx = self.database["molIds"][index][0]
            queryMolecule = self.database["molInputs"][[molIdx], :]
            taskIdx = self.database["taskIds"][index][0]
            # get assay description from taskindex
            label = self.database["labels"][index]

            # Compute support set
            # Collect indices of active and inactive molecules in task
            active_mols_in_task = self.database["dictTaskidActivemolecules"][taskIdx]
            inactive_mols_in_task = self.database["dictTaskidInactivemolecules"][
                taskIdx
            ]
            # Remove query molecule from sets
            if label == True:
                active_mols_in_task = [i for i in active_mols_in_task if i != molIdx]
            else:
                inactive_mols_in_task = [
                    i for i in inactive_mols_in_task if i != molIdx
                ]

            (
                supportSetActives,
                supportSetInactives,
                supportSetActivesSize,
                supportSetInactivesSize,
            ) = self.stratified_splitting_train(
                    active_mols_in_task, inactive_mols_in_task)

            sample = {
                "queryMolecule": queryMolecule,
                "label": label,
                "supportSetActives": supportSetActives,
                "supportSetInactives": supportSetInactives,
                "supportSetActivesSize": supportSetActivesSize,
                "supportSetInactivesSize": supportSetInactivesSize,
                "taskIdx": taskIdx,
            }
            return sample

        def __len__(self):
            return self.len

        def stratified_splitting_train(
            self, active_mols_in_task, inactive_mols_in_task
        ):
            """
            This functions performs a stratified shuffled random split to create the 
            support sets.
            Since within this training loop a query molecule already is given, only a 
            support set has to be returned
            here. So, for every query molecule a support set is drawn.

            The support sets are padded such that the sets are filled up to 12. 
            If you want to use a different data set than FS-Mol, 12 might be the wrong 
            choice.

            :param active_mols_in_task: list of indices
            :param inactive_mols_in_task: list of indices
            """

            # Create labels for molecules
            activeSet_labels = np.ones(len(active_mols_in_task)).reshape(-1, 1)
            inactiveSet_labels = np.zeros(len(inactive_mols_in_task)).reshape(-1, 1)

            # Merge active and inactive molecules
            activeAndInactiveSet = np.array(active_mols_in_task + inactive_mols_in_task)
            activeAndInactiveSet_labels = np.vstack(
                [activeSet_labels, inactiveSet_labels]
            ).reshape(-1)

            # Create splitter object
            skf = StratifiedShuffleSplit(
                n_splits=1, test_size=self.config.supportSet.supportSetSize
            )

            # Split data into query and support set
            _, supportSetSplitIndices = list(
                skf.split(activeAndInactiveSet, activeAndInactiveSet_labels)
            )[0]

            # Prepare support set
            suppportSetActiveAndInactive_ids = activeAndInactiveSet[
                supportSetSplitIndices
            ]
            suppportSetActiveAndInactive_labels = activeAndInactiveSet_labels[
                supportSetSplitIndices
            ]

            supportSetActivesIds = suppportSetActiveAndInactive_ids[
                suppportSetActiveAndInactive_labels == 1
            ]
            supportSetInactivesIds = suppportSetActiveAndInactive_ids[
                suppportSetActiveAndInactive_labels == 0
            ]

            supportSetActives = self.database["molInputs"][supportSetActivesIds, :]
            supportSetInactives = self.database["molInputs"][supportSetInactivesIds, :]

            supportSetActives_size = supportSetActives.shape[0]
            supportSetInactives_size = supportSetInactives.shape[0]

            supportSetActives = np.pad(
                supportSetActives,
                ((0, 12 - supportSetActives_size), (0, 0)),
                "constant",
                constant_values=0,
            )
            supportSetInactives = np.pad(
                supportSetInactives,
                ((0, 12 - supportSetInactives_size), (0, 0)),
                "constant",
                constant_values=0,
            )

            return (
                supportSetActives,
                supportSetInactives,
                supportSetActives_size,
                supportSetInactives_size,
            )

    class _EvalData:
        """
        This is the pytorch dataclass which is required by the validation and test 
        dataloaders.
        - __getitem__ returns a data point in triplet format ( [mol_id, task_id, label] 
        ) and the referring support set
        - __len__ returns the number of training data points
        """

        def __init__(self, database, config):
            self.database = database
            self.config = config

            self.len = len(self.database["query_molIds"])

        def __getitem__(self, index):
            # Get triplet
            molIdx = self.database["query_molIds"][index]
            queryMolecule = self.database["molInputs"][[molIdx], :]
            taskIdx = self.database["query_taskIds"][index]
            label = self.database["query_labels"][index]

            # Get support set
            ## Suport set indices
            supportSetActivesIndices = self.database["supportSetActives"][taskIdx]
            supportSetInactivesIndices = self.database["supportSetInactives"][taskIdx]
            # From indices to support sets
            supportSetActives = self.database["molInputs"][supportSetActivesIndices, :]
            supportSetInactives = self.database["molInputs"][
                supportSetInactivesIndices, :
            ]
            # Support set sizes
            supportSetActivesSize = supportSetActives.shape[0]
            supportSetInactivesSize = supportSetInactives.shape[0]
            # Padded support sets
            supportSetActives = np.pad(
                supportSetActives,
                ((0, 12 - supportSetActivesSize), (0, 0)),
                "constant",
                constant_values=0,
            )
            supportSetInactives = np.pad(
                supportSetInactives,
                ((0, 12 - supportSetInactivesSize), (0, 0)),
                "constant",
                constant_values=0,
            )

            sample = {
                "queryMolecule": queryMolecule,
                "label": label,
                "supportSetActives": supportSetActives,
                "supportSetInactives": supportSetInactives,
                "supportSetActivesSize": supportSetActivesSize,
                "supportSetInactivesSize": supportSetInactivesSize,
                "taskIdx": taskIdx,
                "molIdx": molIdx,
            }
            return sample

        def __len__(self):
            return self.len
