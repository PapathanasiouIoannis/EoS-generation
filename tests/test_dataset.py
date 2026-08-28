from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from eos_generation import ExperimentSettings, plan_experiment
from eos_generation._experiment_planning import _precision_profile
from eos_generation._internal.planning import _selected_groups
from eos_generation.notebook import NotebookSettings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import build_dataset_plots as plots
from test_eos_catalogue import fixture, seal, snapshot, write_csv


class DatasetProfileTests(unittest.TestCase):
    def test_tighter_10_changes_only_grid_tolerances_and_stage_label(self):
        from dataclasses import asdict
        before = _precision_profile('dataset','stellar')
        after = _precision_profile('dataset_10_tighter','stellar')
        self.assertEqual({k:v for k,v in before.items() if k!='tov_stages'},
                         {k:v for k,v in after.items() if k!='tov_stages'})
        self.assertEqual(dict(asdict(before['tov_stages'][0]),name='dataset_10_tighter',
                              sequence_points=10,rtol=1e-11,atol=1e-13),
                         asdict(after['tov_stages'][0]))
        settings = ExperimentSettings.from_values(amplitudes=[0,.12],epsilon_match=80,
            center=450,width=500,ramp_width=350,calculation='stellar',precision='dataset_10_tighter')
        target = ROOT/'runs/test-dataset-10-tighter-no-write'
        self.assertFalse(target.exists())
        with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')), \
             patch('eos_generation.stellar.tov.solve_sequence',side_effect=AssertionError('solver')):
            plan = plan_experiment(settings,output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual((0,0),(plan.to_dict()['scientific_solver_calls'],plan.to_dict()['filesystem_writes']))
        self.assertEqual(20,plan.estimates['sampled_sequence_tidal_targets'])
        self.assertEqual(('none',),plan.child_plans[0].config.requested_plot_groups)
        self.assertIn('10 pressures',plan.summary_text())
        self.assertIn('rtol=1e-11, atol=1e-13',plan.summary_text())
        self.assertIn('not STRICT certification',plan.summary_text())
        self.assertNotEqual(settings.deterministic_hash(),
            ExperimentSettings.from_dict(dict(settings.to_dict(),precision='dataset_20')).deterministic_hash())
        for overrides in ({'calculation':'thermodynamics'}, {'calculation':'stellar','diagnostics':'on'}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision='dataset_10_tighter',**overrides)
        document = json.loads((ROOT/'notebooks/bsk24_dataset.ipynb').read_text())
        guard = next(''.join(c['source']) for c in document['cells'] if ''.join(c['source']).startswith('if CALCULATION'))
        namespace = dict(NotebookSettings=NotebookSettings,AMPLITUDES=[0,.12],EPSILON_MATCH=80,
                         CENTER=450,WIDTH=500,RAMP_WIDTH=350,CALCULATION='stellar',
                         FIXED_MASSES=[1.4],PRECISION='dataset_10_tighter',DIAGNOSTICS='off')
        exec(guard,namespace)
        self.assertEqual(settings,namespace['settings'].to_experiment_settings())

    def test_tight_20_changes_only_sequence_count_and_stage_label(self):
        from dataclasses import asdict
        before = _precision_profile('dataset','stellar')
        after = _precision_profile('dataset_20','stellar')
        self.assertEqual({k:v for k,v in before.items() if k!='tov_stages'},
                         {k:v for k,v in after.items() if k!='tov_stages'})
        self.assertEqual(dict(asdict(before['tov_stages'][0]),name='dataset_20',sequence_points=20),
                         asdict(after['tov_stages'][0]))
        settings = ExperimentSettings.from_values(amplitudes=[0,.12],epsilon_match=80,
            center=450,width=500,ramp_width=350,calculation='stellar',precision='dataset_20')
        target = ROOT/'runs/test-dataset-20-no-write'
        self.assertFalse(target.exists())
        with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')), \
             patch('eos_generation.stellar.tov.solve_sequence',side_effect=AssertionError('solver')):
            plan = plan_experiment(settings,output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual((0,0),(plan.to_dict()['scientific_solver_calls'],plan.to_dict()['filesystem_writes']))
        self.assertEqual(40,plan.estimates['sampled_sequence_tidal_targets'])
        self.assertEqual(('none',),plan.child_plans[0].config.requested_plot_groups)
        self.assertIn('20 pressures',plan.summary_text())
        self.assertIn('not STRICT certification',plan.summary_text())
        self.assertNotEqual(settings.deterministic_hash(),
            ExperimentSettings.from_dict(dict(settings.to_dict(),precision='dataset')).deterministic_hash())
        for overrides in ({'calculation':'thermodynamics'}, {'calculation':'stellar','diagnostics':'on'}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision='dataset_20',**overrides)
        document = json.loads((ROOT/'notebooks/bsk24_dataset.ipynb').read_text())
        guard = next(''.join(c['source']) for c in document['cells'] if ''.join(c['source']).startswith('if CALCULATION'))
        namespace = dict(NotebookSettings=NotebookSettings,AMPLITUDES=[0,.12],EPSILON_MATCH=80,
                         CENTER=450,WIDTH=500,RAMP_WIDTH=350,CALCULATION='stellar',
                         FIXED_MASSES=[1.4],PRECISION='dataset_20',DIAGNOSTICS='off')
        exec(guard,namespace)
        self.assertEqual(settings,namespace['settings'].to_experiment_settings())

    def test_tight_40_changes_only_sequence_count_and_stage_label(self):
        from dataclasses import asdict
        before = _precision_profile('dataset','stellar')
        after = _precision_profile('dataset_40','stellar')
        self.assertEqual({k:v for k,v in before.items() if k!='tov_stages'},
                         {k:v for k,v in after.items() if k!='tov_stages'})
        self.assertEqual(dict(asdict(before['tov_stages'][0]),name='dataset_40',sequence_points=40),
                         asdict(after['tov_stages'][0]))
        self.assertEqual((1e-10,1e-12,1201),
                         (after['tov_stages'][0].rtol,after['tov_stages'][0].atol,after['tov_stages'][0].radial_profile_points))
        settings = ExperimentSettings.from_values(amplitudes=[0,.12],epsilon_match=80,
            center=450,width=500,ramp_width=350,calculation='stellar',precision='dataset_40')
        target = ROOT/'runs/test-dataset-40-no-write'
        self.assertFalse(target.exists())
        with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')), \
             patch('eos_generation.stellar.tov.solve_sequence',side_effect=AssertionError('solver')):
            plan = plan_experiment(settings,output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual((0,0),(plan.to_dict()['scientific_solver_calls'],plan.to_dict()['filesystem_writes']))
        self.assertEqual(2*40,plan.estimates['sampled_sequence_tidal_targets'])
        self.assertEqual(('none',),plan.child_plans[0].config.requested_plot_groups)
        self.assertIn('not STRICT certification',plan.summary_text())
        self.assertIn('40 pressures',plan.summary_text())
        old = ExperimentSettings.from_dict(dict(settings.to_dict(),precision='dataset'))
        self.assertNotEqual(old.deterministic_hash(),settings.deterministic_hash())
        notebook = NotebookSettings.from_values(**settings.to_dict())
        self.assertEqual(settings,notebook.to_experiment_settings())
        for overrides in ({'calculation':'thermodynamics'}, {'calculation':'stellar','diagnostics':'on'}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision='dataset_40',**overrides)
        document = json.loads((ROOT/'notebooks/bsk24_dataset.ipynb').read_text())
        guard = next(''.join(c['source']) for c in document['cells'] if ''.join(c['source']).startswith('if CALCULATION'))
        namespace = dict(NotebookSettings=NotebookSettings,AMPLITUDES=[0,.12],EPSILON_MATCH=80,
                         CENTER=450,WIDTH=500,RAMP_WIDTH=350,CALCULATION='stellar',
                         FIXED_MASSES=[1.4],PRECISION='dataset_40',DIAGNOSTICS='off')
        exec(guard,namespace)
        self.assertEqual(notebook,namespace['settings'])

    def test_relaxed_80_changes_only_sequence_count_and_stage_label(self):
        from dataclasses import asdict
        before = _precision_profile('dataset_relaxed','stellar')
        after = _precision_profile('dataset_relaxed_80','stellar')
        self.assertEqual({k:v for k,v in before.items() if k!='tov_stages'},
                         {k:v for k,v in after.items() if k!='tov_stages'})
        self.assertEqual(dict(asdict(before['tov_stages'][0]),name='dataset_relaxed_80',sequence_points=80),
                         asdict(after['tov_stages'][0]))
        settings = ExperimentSettings.from_values(amplitudes=[0,.12],epsilon_match=80,
            center=450,width=500,ramp_width=350,calculation='stellar',precision='dataset_relaxed_80')
        target = ROOT/'runs/test-dataset-relaxed-80-no-write'
        self.assertFalse(target.exists())
        with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')), \
             patch('eos_generation.stellar.tov.solve_sequence',side_effect=AssertionError('solver')):
            plan = plan_experiment(settings,output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual((0,0),(plan.to_dict()['scientific_solver_calls'],plan.to_dict()['filesystem_writes']))
        self.assertEqual(2*80,plan.estimates['sampled_sequence_tidal_targets'])
        self.assertEqual(('none',),plan.child_plans[0].config.requested_plot_groups)
        self.assertIn('not STRICT certification',plan.summary_text())
        self.assertIn('80 pressures',plan.summary_text())
        old = ExperimentSettings.from_dict(dict(settings.to_dict(),precision='dataset_relaxed'))
        self.assertNotEqual(old.deterministic_hash(),settings.deterministic_hash())
        notebook = NotebookSettings.from_values(**settings.to_dict())
        self.assertEqual(settings,notebook.to_experiment_settings())
        for overrides in ({'calculation':'thermodynamics'}, {'calculation':'stellar','diagnostics':'on'}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision='dataset_relaxed_80',**overrides)
        document = json.loads((ROOT/'notebooks/bsk24_dataset.ipynb').read_text())
        guard = next(''.join(c['source']) for c in document['cells'] if ''.join(c['source']).startswith('if CALCULATION'))
        namespace = dict(NotebookSettings=NotebookSettings,AMPLITUDES=[0,.12],EPSILON_MATCH=80,
                         CENTER=450,WIDTH=500,RAMP_WIDTH=350,CALCULATION='stellar',
                         FIXED_MASSES=[1.4],PRECISION='dataset_relaxed_80',DIAGNOSTICS='off')
        exec(guard,namespace)
        self.assertEqual(notebook,namespace['settings'])

    def test_relaxed_dataset_changes_only_stage_label_and_ode_tolerances(self):
        from dataclasses import asdict
        dataset = _precision_profile("dataset", "stellar")
        relaxed = _precision_profile("dataset_relaxed", "stellar")
        self.assertEqual({k:v for k,v in dataset.items() if k != "tov_stages"},
                         {k:v for k,v in relaxed.items() if k != "tov_stages"})
        before, = dataset['tov_stages']
        after, = relaxed['tov_stages']
        expected = dict(asdict(before), name='dataset_relaxed', rtol=1e-8, atol=1e-10)
        self.assertEqual(expected, asdict(after))
        values = dict(amplitudes=[0,.12],epsilon_match=80,center=450,width=500,
                      ramp_width=350,calculation='stellar')
        original = ExperimentSettings.from_values(**values, precision='dataset')
        candidate = ExperimentSettings.from_values(**values, precision='dataset_relaxed')
        self.assertNotEqual(original.deterministic_hash(),candidate.deterministic_hash())
        target = ROOT / 'runs/test-dataset-relaxed-no-write'
        self.assertFalse(target.exists())
        with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')), \
             patch('eos_generation.stellar.tov.solve_sequence',side_effect=AssertionError('solver')):
            plan = plan_experiment(candidate, output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual((0,0),(plan.to_dict()['scientific_solver_calls'],plan.to_dict()['filesystem_writes']))
        self.assertEqual(2*61,plan.estimates['sampled_sequence_tidal_targets'])
        self.assertEqual(('none',),plan.child_plans[0].config.requested_plot_groups)
        notebook = NotebookSettings.from_values(**candidate.to_dict())
        self.assertEqual(candidate,notebook.to_experiment_settings())
        for overrides in ({'calculation':'thermodynamics'}, {'calculation':'stellar','diagnostics':'on'}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision='dataset_relaxed',**overrides)
        document = json.loads((ROOT/'notebooks/bsk24_dataset.ipynb').read_text())
        guard = next(''.join(c['source']) for c in document['cells']
                     if ''.join(c['source']).startswith('if CALCULATION'))
        namespace = dict(NotebookSettings=NotebookSettings,AMPLITUDES=[0,.12],EPSILON_MATCH=80,
                         CENTER=450,WIDTH=500,RAMP_WIDTH=350,CALCULATION='stellar',
                         FIXED_MASSES=[1.4],PRECISION='dataset_relaxed',DIAGNOSTICS='off')
        exec(guard, namespace)
        self.assertEqual(notebook,namespace['settings'])

    def test_dataset_retains_strict_scientific_settings_except_stellar_grid(self):
        strict = _precision_profile("strict", "stellar")
        dataset = _precision_profile("dataset", "stellar")
        self.assertEqual({k:v for k,v in strict.items() if k != "tov_stages"},
                         {k:v for k,v in dataset.items() if k != "tov_stages"})
        final = strict["tov_stages"][-1]
        single, = dataset["tov_stages"]
        self.assertEqual(("dataset", 61, final.rtol, final.atol, final.radial_profile_points),
                         (single.name, single.sequence_points, single.rtol, single.atol, single.radial_profile_points))
        self.assertEqual([(61,1e-8,1e-10,601),(121,1e-8,1e-10,601),(121,1e-10,1e-12,1201)],
                         [(s.sequence_points,s.rtol,s.atol,s.radial_profile_points) for s in strict["tov_stages"]])
        quick = _precision_profile("quick", "stellar")
        self.assertEqual(17, quick["tov_stages"][0].sequence_points)
        self.assertEqual(9, quick["maximum_mass_initial_points"])

    def test_strict_configuration_hash_includes_a0_owner_contract(self):
        settings = ExperimentSettings.from_values(amplitudes=[0,.04,.08,.12,.16,.2,.24,.28,.32],
            epsilon_match=80,center=200,width=150,ramp_width=125,calculation="stellar",precision="strict")
        plan = plan_experiment(settings, output_root=ROOT / "runs/test-strict-profile-identity")
        self.assertEqual("6e55ab3af25a57e102978ba36a10dfe3d5da760efd68539e7e07745dcdf6bf64",
                         plan.child_plans[0].config.deterministic_hash())

    def test_dataset_plan_is_passive_and_counts_every_control(self):
        target = ROOT / "runs/test-dataset-plan-no-write"
        self.assertFalse(target.exists())
        with patch("eos_generation.stellar.tov.solve_star", side_effect=AssertionError("solver")), \
             patch("eos_generation.stellar.tov.solve_sequence", side_effect=AssertionError("solver")):
            settings = ExperimentSettings.from_values(amplitudes=[-.12,0,.12],epsilon_match=80,
                center=[200,950],width=500,ramp_width=350,calculation="stellar",precision="dataset")
            plan = plan_experiment(settings,output_root=target)
        self.assertFalse(target.exists())
        self.assertEqual(0, plan.to_dict()["scientific_solver_calls"])
        self.assertEqual(0, plan.to_dict()["filesystem_writes"])
        self.assertEqual(5*61, plan.estimates["sampled_sequence_tidal_targets"])
        self.assertTrue(all(c.config.requested_plot_groups == ("none",) for c in plan.child_plans))
        self.assertEqual("dataset", NotebookSettings.from_values(**settings.to_dict()).precision)

    def test_unsupported_dataset_modes_and_mixed_none_fail_closed(self):
        for overrides in ({"calculation":"thermodynamics"}, {"calculation":"stellar","diagnostics":"on"}):
            with self.assertRaises(ValueError):
                ExperimentSettings.from_values(precision="dataset",**overrides)
        self.assertEqual((), _selected_groups(("none",)))
        with self.assertRaises(ValueError):
            _selected_groups(("none","stellar"))

    def test_new_notebook_is_passive_and_uses_public_session(self):
        import nbformat
        from nbclient import NotebookClient
        path = ROOT / "notebooks/bsk24_dataset.ipynb"
        notebook = nbformat.read(path, as_version=4)
        self.assertEqual(["user-settings"], [c.id for c in notebook.cells if c.metadata.get("editable",True)])
        self.assertTrue(all(not c.get("outputs") for c in notebook.cells))
        runs = ROOT / "runs"
        def inventory():
            return {str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in runs.rglob("*") if p.is_file()}
        before = inventory()
        executed = NotebookClient(notebook,timeout=120,kernel_name="python3").execute(cwd=str(ROOT))
        self.assertEqual(before, inventory())
        text = "\n".join(str(o.get("text","")) for c in executed.cells for o in c.get("outputs",[]))
        self.assertIn("EXECUTE_REVIEWED_PLAN=False", text)
        self.assertIn("no per-case stellar refinement envelope", text)


class DatasetPlotTests(unittest.TestCase):
    def test_disabled_plots_keep_population_evidence_and_validator_strict(self):
        from eos_generation._internal.planning import BSk24TrialConfig, BSk24TOVStage, _json_records
        from eos_generation.reporting.plot_orchestration import _actual_plot_inventory
        from eos_generation.reporting._validation_scientific import _validate_response_population_reporting
        from eos_generation.reporting._validation_io import _Layer
        config = BSk24TrialConfig(amplitudes=(0.0,0.12),deltas_mev_fm3=(125.0,),
            fixed_masses_msun=(1.4,),tov_stages=(BSk24TOVStage('dataset',61,1e-10,1e-12,1201),))
        with tempfile.TemporaryDirectory() as tmp:
            packet = Path(tmp)
            rows = [dict(case_id='zero' if a == 0 else 'case',stage='dataset',amplitude=a,
                         delta_mev_fm3=125.0,target_mass_msun=1.4,status='bracketed_and_solved',
                         tidal_status='validated_lambda_validation_v1' if a == 0 else 'failed',
                         k2=0.1,lambda_dimensionless=100.0) for a in (0.0,0.12)]
            pd.DataFrame(rows).to_csv(packet/'fixed_mass_observables.csv',index=False)
            with patch('eos_generation.stellar.tov.solve_star',side_effect=AssertionError('solver')):
                inventory = _actual_plot_inventory(packet,config,groups=('none',))
            self.assertTrue(inventory.status.eq('skipped').all())
            row = inventory.set_index('figure').loc['observable_response_vs_amplitude.png']
            self.assertEqual((2,1,1),tuple(row[k] for k in ('eligible_response_row_count','tidal_validated_count','tidal_omitted_count')))
            inventory.to_csv(packet/'plot_inventory.csv',index=False)
            metadata = {'plot_tidal_completeness':_json_records(inventory.loc[inventory.tidal_completeness_status.ne('not_applicable')])}
            layer = _Layer()
            _validate_response_population_reporting(packet,config.to_dict(),metadata,layer)
            self.assertEqual([],layer.failures)
            # Omitting evidence must still fail the unchanged scientific validator.
            inventory.loc[inventory.figure.eq('observable_response_vs_amplitude.png'),'eligible_response_row_count'] = 0
            inventory.to_csv(packet/'plot_inventory.csv',index=False)
            layer = _Layer()
            _validate_response_population_reporting(packet,config.to_dict(),metadata,layer)
            self.assertTrue(any('response_population_mismatch' in f for f in layer.failures))

    def test_failed_background_and_tidal_rows_break_lines(self):
        rows = pd.DataFrame({"attempted_index":range(5),"is_sampled_peak":[False,False,False,True,False],
            "calculation_status":["success","failed","success","success","success"],
            "tidal_status":["validated_lambda_validation_v1","failed","failed","validated_lambda_validation_v1","validated_lambda_validation_v1"],
            "Mass":[1,2,3,4,5],"Radius":[15,14,13,12,11],"Lambda":[100,90,80,70,60]})
        x,y = plots.sequence_xy(rows,"Radius","Mass")
        self.assertEqual(4,len(x))
        self.assertTrue(np.isnan(x[1]))
        self.assertEqual(3,y[2])
        x,y = plots.sequence_xy(rows,"Mass","Lambda")
        self.assertTrue(np.isnan(y[1:3]).all())
        self.assertEqual(70,y[3])

    def test_exactly_five_figures_deduplication_manifests_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = fixture(root,"dataset",precision="dataset")
            packet = experiment / "geometry_001"
            rows = plots.catalogue.csv_rows(packet / "stellar_sequences.csv")
            for row in rows:
                row.update(k2="0.1",Lambda="100",tidal_status="validated_lambda_validation_v1")
            write_csv(packet / "stellar_sequences.csv",rows)
            seal(packet)
            (experiment / "SHA256SUMS.txt").write_text("synthetic aggregate marker\n")
            destination = experiment.parent / "plots"
            data = experiment.parent / "EOS_DATA"
            before = snapshot(experiment)
            with patch.object(plots.catalogue,"validate_source") as validator, \
                 patch("eos_generation.stellar.tov.solve_star",side_effect=AssertionError("solver")):
                plots.catalogue.build_eos_data(root,experiment,data)
                result = plots.build_dataset_plots(root,experiment,destination,data)
            self.assertEqual(2,validator.call_count)
            self.assertEqual(2,result["unique_eos_count"])
            self.assertEqual(1,result["excluded_rejected_case_occurrence_count"])
            self.assertEqual(5,len(list(destination.glob("*.png"))))
            self.assertFalse(any(p.is_dir() for p in destination.iterdir()))
            self.assertEqual(before,snapshot(experiment))
            self.assertEqual(0,result["solver_calls"])
            self.assertEqual("linear-linear",result["axis_policy"]["eos_pressure"])
            for name,expected in plots.catalogue.manifest(destination).items():
                self.assertEqual(expected,hashlib.sha256((destination/name).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):
                plots.build_dataset_plots(root,experiment,destination,data)
            with patch.object(plots.catalogue,"validate_source",side_effect=ValueError("invalid")):
                with self.assertRaisesRegex(ValueError,"invalid"):
                    plots.build_dataset_plots(root,experiment,experiment.parent/"bad",data)
            self.assertFalse((experiment.parent/"bad").exists())

    def test_dataset_plots_resolve_physical_a0_profile_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = fixture(
                root,
                "physical-a0-dataset",
                precision="dataset_40",
                physical_zero_id="bsk24_baseline_physical",
            )
            packet = experiment / "geometry_001"
            rows = plots.catalogue.csv_rows(packet / "stellar_sequences.csv")
            for row in rows:
                row.update(
                    k2="0.1",
                    Lambda="100",
                    tidal_status="validated_lambda_validation_v1",
                )
            write_csv(packet / "stellar_sequences.csv", rows)
            seal(packet)
            (experiment / "SHA256SUMS.txt").write_text(
                "synthetic aggregate marker\n", encoding="utf-8"
            )
            data = experiment.parent / "EOS_DATA"
            destination = experiment.parent / "plots"
            with patch.object(plots.catalogue, "validate_source"), patch(
                "eos_generation.stellar.tov.solve_star",
                side_effect=AssertionError("solver"),
            ):
                plots.catalogue.build_eos_data(root, experiment, data)
                result = plots.build_dataset_plots(
                    root, experiment, destination, data
                )
            self.assertEqual(2, result["unique_eos_count"])
            index = plots.catalogue.csv_rows(
                destination / "accepted_case_index.csv"
            )
            self.assertEqual(
                {"direct", "case"},
                {row["source_case_id"] for row in index},
            )
            self.assertEqual(0, result["solver_calls"])

    def test_cfl_uses_the_same_five_combined_plot_contract_and_c_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = fixture(
                root, "cfl-dataset", precision="dataset_40", matter_model="cfl"
            )
            packet = experiment / "geometry_001"
            rows = plots.catalogue.csv_rows(packet / "stellar_sequences.csv")
            for row in rows:
                row.update(
                    k2="0.1",
                    Lambda="100",
                    tidal_status="validated_lambda_validation_v1",
                )
            write_csv(packet / "stellar_sequences.csv", rows)
            seal(packet)
            (experiment / "SHA256SUMS.txt").write_text(
                "synthetic aggregate marker\n", encoding="utf-8"
            )
            destination = experiment.parent / "plots"
            data = experiment.parent / "EOS_DATA"
            before = snapshot(experiment)
            with patch.object(plots.catalogue, "validate_source"):
                catalogue_result = plots.catalogue.build_eos_data(
                    root, experiment, data
                )
                plot_result = plots.build_dataset_plots(
                    root, experiment, destination, data
                )
            aliases = plots.catalogue.csv_rows(data / "case_aliases.csv")
            self.assertEqual({"cfl"}, {row["matter_model"] for row in aliases})
            self.assertEqual(
                {"C000000", "C000001"},
                {row["eos_id"] for row in aliases if row["eos_id"]},
            )
            self.assertEqual(["cfl"], catalogue_result["matter_models"])
            self.assertEqual("cfl", plot_result["matter_model"])
            self.assertEqual(5, len(list(destination.glob("*.png"))))
            self.assertEqual(0, plot_result["solver_calls"])
            self.assertEqual(before, snapshot(experiment))

    def test_legacy_bsk_alias_mapping_without_matter_model_still_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = fixture(root, "legacy-plot-data", precision="dataset_40")
            packet = experiment / "geometry_001"
            rows = plots.catalogue.csv_rows(packet / "stellar_sequences.csv")
            for row in rows:
                row.update(
                    k2="0.1",
                    Lambda="100",
                    tidal_status="validated_lambda_validation_v1",
                )
            write_csv(packet / "stellar_sequences.csv", rows)
            seal(packet)
            (experiment / "SHA256SUMS.txt").write_text(
                "synthetic aggregate marker\n", encoding="utf-8"
            )
            data = experiment.parent / "EOS_DATA"
            with patch.object(plots.catalogue, "validate_source"):
                plots.catalogue.build_eos_data(root, experiment, data)
            aliases = plots.catalogue.csv_rows(data / "case_aliases.csv")
            write_csv(
                data / "case_aliases.csv",
                [
                    {key: value for key, value in row.items() if key != "matter_model"}
                    for row in aliases
                ],
            )
            seal(data)
            with patch.object(plots.catalogue, "validate_source"):
                result = plots.build_dataset_plots(
                    root, experiment, experiment.parent / "plots", data
                )
            self.assertEqual("bsk24", result["matter_model"])
            self.assertEqual(5, len(list((experiment.parent / "plots").glob("*.png"))))


if __name__ == "__main__":
    unittest.main()
