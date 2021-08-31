#file structure
mkdir -p for_mike
mkdir -p for_mike/2d
mkdir -p for_mike/4d

#2d
#heatmaps
cp plots/heatmaps/heatmap_nn-regular_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/2d/heatmap_nn-regular_noipcut.pdf
cp plots/heatmaps/heatmap_nn-regular_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/2d/heatmap_nn-regular_withipcut.pdf
cp plots/heatmaps/heatmap_bdt_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/2d/heatmap_bdt_withipcut.pdf
cp plots/heatmaps/heatmap_bdt_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/2d/heatmap_bdt_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf_noipcut.pdf
cp plots/heatmaps/heatmap_nn-one_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-one_withipcut.pdf
cp plots/heatmaps/heatmap_nn-one_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-one_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-oc_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf-oc_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-oc_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf-oc_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-small_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf-small_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-small_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/2d/heatmap_nn-inf-small_noipcut.pdf


#results
cp results/latex/eff_table_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.txt for_mike/2d/eff_table_withipcut.txt
cp results/latex/eff_table_minipchi2+sumpt_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.txt for_mike/2d/eff_table_noipcut.txt

#4d
#heatmaps
cp plots/heatmaps/heatmap_nn-regular_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/heatmap_nn-regular_noipcut.pdf
cp plots/heatmaps/heatmap_nn-regular_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/4d/heatmap_nn-regular_withipcut.pdf
cp plots/heatmaps/heatmap_bdt_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/4d/heatmap_bdt_withipcut.pdf
cp plots/heatmaps/heatmap_bdt_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/heatmap_bdt_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf_noipcut.pdf
cp plots/heatmaps/heatmap_nn-one_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-one_withipcut.pdf
cp plots/heatmaps/heatmap_nn-one_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-one_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-oc_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-oc_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-oc_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-oc_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-small_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-small_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-small_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-small_noipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-mon-vchi2_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-mon-vchi2_withipcut.pdf
cp plots/heatmaps/heatmap_nn-inf-mon-vchi2_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/heatmap_nn-inf-mon-vchi2_noipcut.pdf

#results
cp results/latex/eff_table_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.txt for_mike/4d/eff_table_withipcut.txt
cp results/latex/eff_table_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.txt for_mike/4d/eff_table_noipcut.txt

#violins
cp plots/violins/violins_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/violins_withipcut.pdf
cp plots/violins/violins_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/violins_noipcut.pdf

#rate_vs_eff
cp plots/scatter/rate_vs_eff_nn-regular_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/rate_vs_eff_nn-regular_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-regular_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/4d/rate_vs_eff_nn-regular_withipcut.pdf
cp plots/scatter/rate_vs_eff_bdt_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_None_None_None.pdf for_mike/4d/rate_vs_eff_bdt_withipcut.pdf
cp plots/scatter/rate_vs_eff_bdt_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/rate_vs_eff_bdt_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf_withipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-one_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-one_withipcut.pdf
cp plots/scatter/rate_vs_eff_nn-one_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-one_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-oc_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-oc_withipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-oc_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-oc_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-small_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-small_withipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-small_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-small_noipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-mon-vchi2_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:6_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-mon-vchi2_withipcut.pdf
cp plots/scatter/rate_vs_eff_nn-inf-mon-vchi2_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/rate_vs_eff_nn-inf-mon-vchi2_noipcut.pdf

#eff vs kinematics
cp plots/scatter/eff_vs_kinematics_bdt_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/eff_vs_kinematics_bdt_nopicut.pdf
cp plots/scatter/eff_vs_kinematics_nn-inf_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_max-norm_direct_vector.pdf for_mike/4d/eff_vs_kinematics_nn-inf_nopicut.pdf
cp plots/scatter/eff_vs_kinematics_nn-regular_fdchi2+sumpt+vchi2+minipchi2_lhcb_unnormed_heavy-flavor_svPT:1000+trkPT:200+svchi2:20+ipcuttrain:10_None_None_None.pdf for_mike/4d/eff_vs_kinematics_nn-regular_nopicut.pdf

tar -cvzf for_mike.tar.gz for_mike

rm -r for_mike
