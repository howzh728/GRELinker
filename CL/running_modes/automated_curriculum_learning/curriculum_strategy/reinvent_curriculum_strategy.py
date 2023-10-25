import time
import sys 
sys.path.append("../../..") 
from typing import List, Tuple
from parameters.constants import constants as C
import util


import generate
import analyze as anal
import numpy as np
import torch
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.diversity_filter import DiversityFilter
from reinvent_scoring.scoring.diversity_filters.curriculum_learning.update_diversity_filter_dto import \
    UpdateDiversityFilterDTO
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.automated_curriculum_learning.actions.reinvent_sample_model import ReinventSampleModel
from running_modes.automated_curriculum_learning.curriculum_strategy.base_curriculum_strategy import \
    BaseCurriculumStrategy
from running_modes.automated_curriculum_learning.dto import SampledBatchDTO, CurriculumOutcomeDTO, TimestepDTO

   



class ReinventCurriculumStrategy(BaseCurriculumStrategy):

    def get_ts_properties(self):
        """ Loads the training sets properties from CSV as a dictionary, properties
        are used later for model evaluation.
        """
        filename = self.C.training_set[:-3] + "csv"
        self.ts_properties = util.load_ts_properties(csv_path=filename)


    def run(self) -> CurriculumOutcomeDTO:
        self.C = C
        step_counter = 0
        self.disable_prior_gradients()

        for item_id, sf_configuration in enumerate(self._parameters.curriculum_objectives):
            start_time = time.time()
            scoring_function = self._setup_scoring_function(item_id)
            step_counter = self.promote_agent(agent=self._agent, scoring_function=scoring_function,
                                              step_counter=step_counter, start_time=start_time,
                                              merging_threshold=sf_configuration.score_threshold,prior=self._prior,optimizer=self._optimizer,
                                              )
            self.save_and_flush_memory(agent=self._agent, memory_name=f"_merge_{item_id}")
        is_successful_curriculum = step_counter < self._parameters.max_num_iterations
        outcome_dto = CurriculumOutcomeDTO(self._agent, step_counter, successful_curriculum=is_successful_curriculum)

        return outcome_dto

    

    def take_step(self, agent: GenerativeModelBase, scoring_function: BaseScoringFunction,
                  step:int, start_time: float,prior,optimizer) -> float:

        
        #generate problems
        self.start_time = time.time()
        self.get_ts_properties()
        g, a, p, t ,two_idx_agent,two_idx_prior= generate.build_graphs(agent_model=agent,
                                           prior_model=prior,
                                           n_graphs_to_generate=self.C.batch_size,
                                           batch_size=self.C.batch_size,
                                           mols_too=True
                                          )

        smi_list = []
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import random
        validity_tensor = torch.zeros(len(g[0]), device="cuda")

        for idxm, molecular_graph in enumerate(g[0]):
            try:
                mol = molecular_graph.get_molecule()
                smi = Chem.MolToSmiles(mol)
                smi_list.append(smi)
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(mol)
            except (ValueError, RuntimeError, AttributeError):
                smi_list.append('C')
        connect_smi_list = []
        for id,atom_idx in enumerate(two_idx):
            try:
                fragments_smi = self.C.generate_fragments
                fragments = Chem.MolFromSmiles(fragments_smi)
                linker_smi_1 = smi_list[id]
                
                linker = Chem.MolFromSmiles(linker_smi_1)
                #connect
                combo = Chem.CombineMols(fragments,linker)
                idx_list = []
                for atom in fragments.GetAtoms():
                    if atom.GetSymbol() == '*':
                        f2 = atom.GetIdx()
                        idx_list.append(f2)

                du = Chem.MolFromSmiles('*')
                combo = AllChem.DeleteSubstructs(combo,du)
                edcombo = Chem.EditableMol(combo)
                try:
                    edcombo.AddBond(idx_list[0],atom_idx[0]+idx_list[1]-2,order=Chem.rdchem.BondType.SINGLE)
                    edcombo.AddBond(idx_list[1]-2,atom_idx[1]+idx_list[1]-2,order=Chem.rdchem.BondType.SINGLE)
                    final_connect_mol = edcombo.GetMol()
                except:
            
                    fragment_smi_0 = fragments_smi.split('.')[0]
                    fragment_smi_1 = fragments_smi.split('.')[1]
                    
                    match = '****sfs'
                    for _ in range(1000):
                        if not Chem.MolFromSmiles(match):
                            s = linker_smi_1
                            inStr = "(*)"
                            for i in range(2):
                                index = random.randint(0, len(s))
                                s = "".join([s[:index], inStr, s[index:]])
                            match = s
                        else:
                            break
                    smi_linker = []
                    for _ in range(1000):
                        smi_link = Chem.MolToSmiles(Chem.MolFromSmiles(s), doRandom=True)
                        if smi_link[0] == "*" and smi_link[-1] == '*':
                            smi_linker.append(smi_link)
                    
                        
                    final_smi = (fragment_smi_1 + smi_linker[0] + fragment_smi_0).replace("*", "")

                    final_connect_mol = Chem.MolFromSmiles(final_smi)
                    
                final_connect_smi = Chem.MolToSmiles(final_connect_mol)
                validity_tensor[idx] = 1

            except:
                fragment_smi_0 = fragments_smi.split('.')[0]
                fragment_smi_1 = fragments_smi.split('.')[1]
                    
                final_connect_smi = (fragment_smi_1 + fragment_smi_0).replace("*", "")
            connect_smi_list.append(final_connect_smi)
            

        
            

                
            
            





        # analyze properties of new graphs and save results
        validity_linker_tensor, linker_smiles = anal.evaluate_generated_graphs(generated_graphs=g[0],
                                                                termination=t,
                                                                agent_lls=a,
                                                                prior_lls=p,
                                                                start_time=self.start_time,
                                                                ts_properties=self.ts_properties,
                                                                generation_batch_idx=idx)

        uniqueness_tensor = util.get_unique_tensor(connect_smi_list)
        # score = compute_score(g[1], t, validity_tensor, uniqueness_tensor, connect_smi_list, self.drd2_model)
        
        
        
        score, score_summary = self._scoring(scoring_function, connect_smi_list, step)
        score_tensor = torch.tensor(score)
        augmented_likelihood = p + C.sigma*score_tensor
        loss = (1-self.C.alpha) * torch.mean(self.compute_loss(score_tensor, a, p, uniqueness_tensor))  

        self._logging(agent=agent, start_time=self.start_time, step=step,
                      score_summary=score_summary, agent_likelihood=a,
                      prior_likelihood=p, augmented_likelihood=augmented_likelihood)

        score_write = torch.mean(torch.clone(score_tensor)).item()
        loss_write = torch.clone(loss)
        a_write = torch.clone(a)
        p_write = torch.clone(p)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        score = score.mean()
        return score

    def _sampling(self, agent) -> SampledBatchDTO:
        sampling_action = ReinventSampleModel(agent, self._parameters.batch_size, self._logger)
        sampled_sequences = sampling_action.run()
        return sampled_sequences

    def _scoring(self, scoring_function, smiles: List[str], step) -> Tuple[np.ndarray, FinalSummary] :
        score_summary = scoring_function.get_final_score_for_step(smiles, step)
        dto = UpdateDiversityFilterDTO(score_summary, [], step)
        score = self._diversity_filter.update_score(dto)
        return score, score_summary

    def _updating(self, sampled, score, inception, agent):
        agent_likelihood, prior_likelihood, augmented_likelihood = \
            self.learning_strategy.run(sampled, score, inception, agent)
        return agent_likelihood, prior_likelihood, augmented_likelihood

    def _logging(self, agent: GenerativeModelBase, start_time: float, step: int, score_summary: FinalSummary,
                  agent_likelihood: torch.tensor, prior_likelihood: torch.tensor, augmented_likelihood: torch.tensor):
        report_dto = TimestepDTO(start_time, self._parameters.max_num_iterations, step, score_summary,
                                 agent_likelihood, prior_likelihood, augmented_likelihood)
        self._logger.timestep_report(report_dto, self._diversity_filter, agent)

    def save_and_flush_memory(self, agent, memory_name: str):
        self._logger.save_merging_state(agent, self._diversity_filter, name=memory_name)
        self._diversity_filter = DiversityFilter(self._parameters.diversity_filter)

    def compute_loss(self,score, agent_ll, prior_ll, uniqueness_tensor):


        augmented_prior_ll = prior_ll + C.sigma*score


        difference = agent_ll - augmented_prior_ll
        loss = difference*difference

        mask = (uniqueness_tensor != 0).int()
        loss = loss * mask

        return loss

