#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright (C) 2024 Max Planck Institute for Multidisclplinary Sciences
# Copyright (C) 2024 Bharti Arora <bharti.arora@mpinat.mpg.de>
# Copyright (C) 2024 Ajinkya Kulkarni <ajinkya.kulkarni@mpinat.mpg.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

########################################################################################

# Run from the project root: streamlit run app/streamlit_app.py

########################################################################################

import sys
import os
sys.dont_write_bytecode = True

# Ensure src/ is on the path regardless of where streamlit is invoked from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 2PLSM modules (no heavy ML dependency)
from modules_2photon import (
	make_filtered_image,
	convolve,
	make_image_gradients,
	make_structure_tensor_2d,
	make_coherence,
	make_orientation,
	percentage_area,
	perform_statistical_analysis,
	load_pandas_dataframe,
)

########################################################################################
# Page config

st.set_page_config(
	page_title="MALDI-MSI · 2PLSM · Histology",
	page_icon="🔬",
	layout="wide",
	initial_sidebar_state="collapsed",
)

########################################################################################
# Header

st.title("MALDI-MSI · 2PLSM · Histology Analysis")
st.markdown(
	"*Arora et al., npj Imaging 2024 · "
	"[DOI: 10.1038/s44303-024-00041-3](https://doi.org/10.1038/s44303-024-00041-3)*"
)
st.divider()

########################################################################################
# Shared helpers

def _read_upload(uploaded_file):
	"""Convert a Streamlit UploadedFile to a numpy array."""
	return np.array(Image.open(io.BytesIO(uploaded_file.read())))


def _show_fig(fig, caption=None):
	"""Display a matplotlib figure in Streamlit then close it."""
	st.pyplot(fig, use_container_width=True)
	if caption:
		st.caption(caption)
	plt.close(fig)


########################################################################################
# StarDist model — loaded once per session, cached across reruns

@st.cache_resource(show_spinner="Loading StarDist model (first run only)…")
def _load_stardist():
	from stardist.models import StarDist2D
	return StarDist2D.from_pretrained("2D_versatile_he")


########################################################################################
# Tabs

tab1, tab2, tab3 = st.tabs([
	"🔬 Collagen Texture (2PLSM)",
	"🧫 Nuclei Segmentation (H&E)",
	"📂 Batch Processing",
])

########################################################################################
# TAB 1 — Collagen Texture Analysis
########################################################################################

with tab1:
	st.header("Collagen Fibre Texture Analysis")
	st.markdown(
		"Upload a 2PLSM SHG image (grayscale TIFF). "
		"Computes local coherence, fibre orientation, and collagen density."
	)

	col_up, col_params = st.columns([1, 1], gap="large")

	with col_up:
		t1_file = st.file_uploader(
			"Upload SHG image (.tif / .tiff)",
			type=["tif", "tiff"],
			key="t1_file",
		)

	with col_params:
		with st.expander("Parameters", expanded=True):
			t1_sigma    = st.slider("Gaussian filter sigma",        1,  5,  2,   key="t1_sigma")
			t1_density  = st.slider("Local density window (px)",    5,  50, 20,  key="t1_density")
			t1_struct   = st.slider("Structure tensor sigma",       5,  20, 10,  key="t1_struct")
			t1_dthresh  = st.slider("Density threshold (fibrotic)", 0.1, 0.9, 0.5, step=0.05, key="t1_dthresh")

	if t1_file is not None:
		if st.button("Run Texture Analysis", type="primary", key="t1_run"):
			with st.spinner("Analysing collagen texture…"):

				raw = _read_upload(t1_file).astype(np.float32)
				if raw.ndim == 3:
					raw = raw[:, :, 0]
				denom = raw.max() - raw.min()
				raw = 255.0 * (raw - raw.min()) / (denom if denom > 0 else 1.0)

				threshold = max(int(np.median(raw)), 2)
				filtered  = make_filtered_image(raw, t1_sigma)

				k = int(t1_density)
				k = k + 1 if k % 2 == 0 else k
				k = max(k, 3)
				kernel  = np.ones((k, k), dtype=np.float32) / (k * k)
				density = convolve(raw, kernel)
				density = np.divide(
					density, density.max(),
					out=np.full_like(density, np.nan),
					where=density.max() != 0,
				)

				dens_t = density.copy()
				dens_t[dens_t < t1_dthresh] = np.nan
				fibrotic_pct = percentage_area(dens_t)

				gx, gy = make_image_gradients(filtered)
				ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, t1_struct)
				coherence   = make_coherence(filtered, EV, ST, threshold)
				orientation = make_orientation(filtered, Jxx, Jxy, Jyy, threshold)

				results = perform_statistical_analysis(
					t1_file.name, t1_struct, None, coherence, fibrotic_pct)
				df = load_pandas_dataframe(results)
				df.insert(0, "Filename", results[:, 0])

			st.success("Analysis complete.")

			c1, c2, c3, c4 = st.columns(4)
			for col, img, title, cmap, vmin, vmax in [
				(c1, raw,         "Raw SHG",        "gray",    0,   255),
				(c2, density,     "Local Density",   "inferno", 0,   1),
				(c3, coherence,   "Coherence",       "inferno", 0,   1),
				(c4, orientation, "Orientation (°)", "hsv",     0,   180),
			]:
				with col:
					fig, ax = plt.subplots(figsize=(4, 4))
					ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
					ax.set_title(title, fontsize=10)
					ax.axis("off")
					_show_fig(fig)

			st.subheader("Statistics")
			st.dataframe(df, use_container_width=True)
			st.download_button(
				"Download CSV",
				df.to_csv(index=False),
				file_name=f"{t1_file.name}_texture_results.csv",
				mime="text/csv",
				key="t1_download",
			)

########################################################################################
# TAB 2 — Nuclei Segmentation
########################################################################################

with tab2:
	st.header("Nuclei Segmentation & Density Analysis")
	st.markdown(
		"Upload an H&E stained histology image (RGB TIFF). "
		"Detects nuclei using StarDist2D and maps nuclear distribution density."
	)

	col_up2, col_p2 = st.columns([1, 1], gap="large")

	with col_up2:
		t2_file = st.file_uploader(
			"Upload H&E image (.tif / .tiff)",
			type=["tif", "tiff"],
			key="t2_file",
		)
		t2_normalise = st.checkbox(
			"Apply stain normalisation (Ruifrok & Johnston)",
			value=False,
			key="t2_norm",
		)

	with col_p2:
		with st.expander("Parameters", expanded=True):
			t2_prob  = st.slider("Detection probability threshold", 0.10, 0.90, 0.40, step=0.05, key="t2_prob")
			t2_nms   = st.slider("NMS threshold",                   0.10, 0.50, 0.30, step=0.05, key="t2_nms")
			t2_nl    = st.slider("Normalisation percentile (low)",  1,    40,   25,              key="t2_nl")
			t2_nh    = st.slider("Normalisation percentile (high)", 60,   99,   85,              key="t2_nh")

	if t2_file is not None:
		if st.button("Segment Nuclei", type="primary", key="t2_run"):

			try:
				model = _load_stardist()
			except Exception as exc:
				st.error(
					f"StarDist or TensorFlow is not installed: {exc}\n\n"
					"Install with: `pip install stardist tensorflow`"
				)
				st.stop()

			with st.spinner("Segmenting nuclei…"):
				from modules_histo import (
					normalize_density_maps,
					mean_filter,
					MyNormalizer,
					normalize_staining,
				)

				img = _read_upload(t2_file)

				if t2_normalise:
					img = normalize_staining(img)

				mi = np.percentile(img, t2_nl)
				ma = np.percentile(img, t2_nh)
				normalizer = MyNormalizer(mi, ma)

				labels, _ = model.predict_instances(
					img,
					prob_thresh=t2_prob,
					nms_thresh=t2_nms,
					normalizer=normalizer,
					n_tiles=(2, 2, 1),
				)
				labels = labels.astype(np.int32)

				local_density = normalize_density_maps(
					mean_filter(labels).astype(np.float32)
				)

				n_nuclei = int(labels.max())
				pct_high = (
					float(np.mean(local_density[labels > 0] >= 0.5) * 100)
					if n_nuclei > 0 else 0.0
				)
				stats_df = pd.DataFrame([{
					"Filename":                   t2_file.name,
					"n_nuclei":                   n_nuclei,
					"% High density regions":     round(pct_high, 2),
					"% Low density regions":      round(100.0 - pct_high, 2),
				}])

			st.success(f"Found **{n_nuclei}** nuclei.")

			c1, c2, c3 = st.columns(3)
			for col, disp_img, title, cmap in [
				(c1, img,           "H&E Image",       None),
				(c2, labels,        "Nuclei Labels",   "nipy_spectral"),
				(c3, local_density, "Nuclear Density",  "inferno"),
			]:
				with col:
					fig, ax = plt.subplots(figsize=(4, 4))
					ax.imshow(disp_img, cmap=cmap)
					ax.set_title(title, fontsize=10)
					ax.axis("off")
					_show_fig(fig)

			st.subheader("Statistics")
			st.dataframe(stats_df, use_container_width=True)
			st.download_button(
				"Download CSV",
				stats_df.to_csv(index=False),
				file_name=f"{t2_file.name}_nuclei_results.csv",
				mime="text/csv",
				key="t2_download",
			)

########################################################################################
# TAB 3 — Batch Processing
########################################################################################

with tab3:
	st.header("Batch Processing")
	st.markdown(
		"Upload multiple images and run the same analysis on each. "
		"Individual image figures are not shown — only the aggregated statistics table."
	)

	t3_type = st.radio(
		"Analysis type",
		["Collagen Texture (2PLSM)", "Nuclei Segmentation (H&E)"],
		horizontal=True,
		key="t3_type",
	)

	t3_files = st.file_uploader(
		"Upload images (.tif / .tiff)",
		type=["tif", "tiff"],
		accept_multiple_files=True,
		key="t3_files",
	)

	if t3_type == "Collagen Texture (2PLSM)":
		with st.expander("Parameters", expanded=False):
			b_sigma   = st.slider("Gaussian filter sigma",        1,  5,  2,   key="b_sigma")
			b_density = st.slider("Local density window (px)",    5,  50, 20,  key="b_density")
			b_struct  = st.slider("Structure tensor sigma",       5,  20, 10,  key="b_struct")
			b_dthresh = st.slider("Density threshold (fibrotic)", 0.1, 0.9, 0.5, step=0.05, key="b_dthresh")
	else:
		with st.expander("Parameters", expanded=False):
			b_prob  = st.slider("Probability threshold",         0.10, 0.90, 0.40, step=0.05, key="b_prob")
			b_nms   = st.slider("NMS threshold",                 0.10, 0.50, 0.30, step=0.05, key="b_nms")
			b_nl    = st.slider("Normalisation percentile (low)", 1,   40,   25,              key="b_nl")
			b_nh    = st.slider("Normalisation percentile (high)",60,  99,   85,              key="b_nh")

	if t3_files and st.button("Run Batch", type="primary", key="t3_run"):

		if t3_type == "Nuclei Segmentation (H&E)":
			try:
				batch_model = _load_stardist()
			except Exception as exc:
				st.error(f"StarDist or TensorFlow not installed: {exc}")
				st.stop()

		progress_bar = st.progress(0)
		status_text  = st.empty()
		all_rows     = []
		n_total      = len(t3_files)

		for i, uf in enumerate(t3_files):
			status_text.text(f"Processing {uf.name}  ({i + 1} / {n_total})")
			try:
				raw = _read_upload(uf).astype(np.float32)

				if t3_type == "Collagen Texture (2PLSM)":
					if raw.ndim == 3:
						raw = raw[:, :, 0]
					denom = raw.max() - raw.min()
					raw = 255.0 * (raw - raw.min()) / (denom if denom > 0 else 1.0)
					threshold = max(int(np.median(raw)), 2)
					filtered  = make_filtered_image(raw, b_sigma)
					k = int(b_density)
					k = k + 1 if k % 2 == 0 else k
					k = max(k, 3)
					kernel  = np.ones((k, k), dtype=np.float32) / (k * k)
					density = convolve(raw, kernel)
					density = np.divide(
						density, density.max(),
						out=np.full_like(density, np.nan),
						where=density.max() != 0,
					)
					dens_t = density.copy()
					dens_t[dens_t < b_dthresh] = np.nan
					fibrotic_pct = percentage_area(dens_t)
					gx, gy = make_image_gradients(filtered)
					ST, EV, EVec, Jxx, Jxy, Jyy = make_structure_tensor_2d(gx, gy, b_struct)
					coherence = make_coherence(filtered, EV, ST, threshold)
					res   = perform_statistical_analysis(uf.name, b_struct, None, coherence, fibrotic_pct)
					df_row = load_pandas_dataframe(res)
					df_row.insert(0, "Filename", res[:, 0])
					all_rows.append(df_row)

				else:
					from modules_histo import (
						normalize_density_maps, mean_filter, MyNormalizer
					)
					img = raw.astype(np.uint8)
					mi  = np.percentile(img, b_nl)
					ma  = np.percentile(img, b_nh)
					normalizer = MyNormalizer(mi, ma)
					labels, _ = batch_model.predict_instances(
						img,
						prob_thresh=b_prob,
						nms_thresh=b_nms,
						normalizer=normalizer,
						n_tiles=(2, 2, 1),
					)
					labels   = labels.astype(np.int32)
					n_nuclei = int(labels.max())
					ld = normalize_density_maps(mean_filter(labels).astype(np.float32))
					pct_high = (
						float(np.mean(ld[labels > 0] >= 0.5) * 100)
						if n_nuclei > 0 else 0.0
					)
					all_rows.append(pd.DataFrame([{
						"Filename":               uf.name,
						"n_nuclei":               n_nuclei,
						"% High density regions": round(pct_high, 2),
						"% Low density regions":  round(100.0 - pct_high, 2),
					}]))

			except Exception as exc:
				st.warning(f"Skipped **{uf.name}**: {exc}")

			progress_bar.progress((i + 1) / n_total)

		status_text.text("Done.")

		if all_rows:
			summary = pd.concat(all_rows, ignore_index=True)
			st.subheader("Batch Results")
			st.dataframe(summary, use_container_width=True)
			st.download_button(
				"Download Batch CSV",
				summary.to_csv(index=False),
				file_name="batch_results.csv",
				mime="text/csv",
				key="t3_download",
			)
		else:
			st.warning("No files were processed successfully.")
