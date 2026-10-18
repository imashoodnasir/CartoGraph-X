param(
  [string]$Root = 'D:\Research\CartoGen',
  [string]$Destination = 'D:\Research\CartoGen\CartoGraphX_Publication_Figures'
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
$Destination = [IO.Path]::GetFullPath($Destination)
$base = Join-Path $Root 'data\processed\cross_country'
$records = [System.Collections.Generic.List[object]]::new()
$folders = @('01_Methodology','02_Main_Results','03_Supplementary','04_Additional_Maps','05_Review_Alternatives')
foreach ($folder in $folders) { New-Item -ItemType Directory -Force -Path (Join-Path $Destination $folder) | Out-Null }

function Copy-Pair {
  param([string]$Group,[string]$Tag,[string]$SourceStem,[string]$DestStem)
  $destDir = Join-Path $Destination $Group
  New-Item -ItemType Directory -Force -Path $destDir | Out-Null
  foreach ($ext in @('.png','.pdf')) {
    $src = $SourceStem + $ext
    $target = Join-Path $destDir ($DestStem + $ext)
    $status = 'MISSING'
    if (Test-Path -LiteralPath $src -PathType Leaf) {
      Copy-Item -LiteralPath $src -Destination $target -Force
      $status = 'COPIED'
    }
    $records.Add([pscustomobject]@{Section=$Group;Label=$Tag;Format=$ext.TrimStart('.');Status=$status;Source=$src;Destination=$target})
  }
}
function Add-Set {
  param([string]$Group,[string]$Tag,[string]$Folder,[string[]]$Names)
  foreach ($name in $Names) {
    Copy-Pair $Group $Tag (Join-Path $Folder $name) $name
  }
}
$pub = Join-Path $base 'cartographx_v2_final_publication_package_v1\figures'
$method = Join-Path $base 'cartographx_v2_methodology_figure_v1'
$general = Join-Path $base 'domain_generalization\figures'
$maps = Join-Path $base 'cartographx_maps'
$generated = Join-Path $Root 'figures_5_9'

# METHODOLOGY: architecture, graph, real-world road context
Add-Set '01_Methodology' 'Architecture and graph construction' $method @(
 'Fig03_CartoGraphX_Technical_Exact_V2',
 'CartoGraphX_Methodology_RealMaps_vilnius_central',
 'Fig02_Real_Road_Graph_warsaw',
 'Fig02_TemplateMatched_RealData_warsaw'
)
Add-Set '01_Methodology' 'Six-city data and prediction atlas' $pub @('F02_six_city_source_reference_prediction_atlas')

# MAIN RESULTS: quantitative protocol results and transfer
Add-Set '02_Main_Results' 'Baseline comparison' $pub @('F03_six_protocol_discrimination')
Add-Set '02_Main_Results' 'Transfer matrix and zero-shot gap' $general @(
 'figure_03_cartographx_transfer_matrix','figure_04_zero_shot_generalization_gap'
)
Add-Set '02_Main_Results' 'Architectural ablation' $pub @('F06_architecture_ablation_heatmap')
Add-Set '02_Main_Results' 'Graph sensitivity' $generated @('Fig05_Graph_Construction_Sensitivity')
Add-Set '02_Main_Results' 'Training fraction sensitivity' $generated @('Fig06_Training_Fraction_Learning_Curves')
Add-Set '02_Main_Results' 'Spatial tile robustness' $generated @(
 'Fig07A_Spatial_Tile_MCC_Heatmaps','Fig07B_Spatial_Tile_MCC_Heatmaps','Fig07C_Spatial_Tile_MCC_Heatmaps'
)
Add-Set '02_Main_Results' 'Matched-seed statistical comparison' $generated @('Fig08_Paired_MCC_Forest_Plot')
Add-Set '02_Main_Results' 'Accuracy latency tradeoff' $generated @('Fig09_Accuracy_Latency_Tradeoff')
Add-Set '02_Main_Results' 'City robustness' $pub @('F09_city_geographic_robustness')

# MAIN RESULTS: representative country-specific road-network comparisons
foreach ($entry in @(@('lithuania','vilnius'),@('poland','warsaw'))) {
  $country=$entry[0]; $city=$entry[1]
  $dir = Join-Path $maps "protocol_comparisons\$country\$city"
  Copy-Pair '02_Main_Results' "Representative $city protocol comparison" (Join-Path $dir "${city}_within_zero_joint_comparison") "Map_${city}_within_zero_joint_comparison"
}

# SUPPLEMENT: full city comparisons and zero-shot, joint and within-country road maps
foreach ($entry in @(
  @('lithuania','kaunas'),@('lithuania','klaipeda'),@('lithuania','vilnius'),
  @('poland','gdansk'),@('poland','krakow'),@('poland','warsaw')
)) {
  $country=$entry[0]; $city=$entry[1]
  $dir = Join-Path $maps "protocol_comparisons\$country\$city"
  foreach ($suffix in @('within_zero_joint_comparison','representative_tile_protocol_comparison')) {
    $group = if ($city -in @('vilnius','warsaw') -and $suffix -eq 'within_zero_joint_comparison') {'05_Review_Alternatives'} else {'03_Supplementary'}
    Copy-Pair $group "Protocol and tile comparisons: $city" (Join-Path $dir "${city}_$suffix") "${city}_$suffix"
  }
  $dir = Join-Path $maps "cities\$country\$city"
  foreach ($regime in @('zero_shot','joint_country','within_country')) {
    foreach ($suffix in @('four_panel','probability')) {
      Copy-Pair '03_Supplementary' "Road prediction maps: $city" (Join-Path $dir "${city}_${regime}_${suffix}") "${city}_${regime}_${suffix}"
    }
  }
  $dir = Join-Path $maps "tiles\$country\$city"
  $tileFiles = @(Get-ChildItem -LiteralPath $dir -File -Filter '*.png' -ErrorAction SilentlyContinue)
  foreach ($f in $tileFiles) {
    Copy-Pair '04_Additional_Maps' "Selected tile: $city" (Join-Path $dir $f.BaseName) $f.BaseName
  }
}

# SUPPLEMENT: domain generalization, calibration, error profiles, cartographic fidelity
Add-Set '03_Supplementary' 'Generalization diagnostics' $general @(
 'figure_01_mcc_by_protocol','figure_02_balanced_accuracy_by_protocol',
 'figure_05_joint_country_gain','figure_06_cartographx_generalization_regimes'
)
Add-Set '03_Supplementary' 'Reliability and error profiles' $pub @(
 'F04_reliability_diagrams','F10_false_positive_negative_profiles'
)
$quality = Join-Path $base 'cartographic_quality\figures'
Add-Set '03_Supplementary' 'Cartographic preservation' $quality @(
 'figure_cartographic_preservation_city_heatmap',
 'figure_connectivity_preservation_city_heatmap',
 'figure_coverage_jaccard_city_heatmap'
)
$shift = Join-Path $base 'domain_shift\figures'
Add-Set '03_Supplementary' 'Transfer degradation' $shift @('figure_08_transfer_degradation')

# REVIEW ALTERNATIVES: earlier publication-package variants (not necessarily final)
Add-Set '05_Review_Alternatives' 'Previously generated figure versions' $pub @(
 'F05_domain_shift_asymmetry','F07_training_fraction_sensitivity',
 'F08_graph_topology_sensitivity'
)
foreach ($city in @('gdansk','kaunas_central','klaipeda_central','krakow','vilnius_central','warsaw')) {
  Copy-Pair '05_Review_Alternatives' "Single-city map: $city" (Join-Path $pub "F01_map_$city") "F01_map_$city"
}

$manifest = Join-Path $Destination 'Publication_Figure_Manifest.csv'
$records | Export-Csv -LiteralPath $manifest -NoTypeInformation -Encoding UTF8
$copied = @($records | Where-Object Status -eq 'COPIED')
$missing = @($records | Where-Object Status -eq 'MISSING')
$missing | Export-Csv -LiteralPath (Join-Path $Destination 'Missing_Figures.csv') -NoTypeInformation -Encoding UTF8
Write-Host "`nDESTINATION: $Destination"
Write-Host "COPIED: $($copied.Count) files | MISSING: $($missing.Count) files"
Write-Host "MANIFEST: $manifest"
Write-Host "`nCopied by folder:"
$copied | Group-Object Section | Sort-Object Name | Format-Table Count,Name -AutoSize
if ($missing.Count -gt 0) {
 Write-Warning 'Some files are missing. See Missing_Figures.csv. No substitute images were created.'
 $missing | Select-Object -First 15 Section,Source | Format-Table -Wrap
}
