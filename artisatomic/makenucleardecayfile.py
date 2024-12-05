#!/usr/bin/env python3
import io
import math
from pathlib import Path

import artistools as at
import numpy as np
import pandas as pd
import polars as pl
import requests


def main():
    PYDIR = Path(__file__).parent.resolve()
    atomicdata = pd.read_csv(PYDIR / "atomic_properties.txt", sep=r"\s+", comment="#")
    elsymbols = ["n", *list(atomicdata["symbol"].values)]

    outfolder = Path(__file__).parent.parent.absolute() / "artis_files" / "data"
    outfolder.mkdir(parents=True, exist_ok=True)
    pd.options.display.max_rows = 999
    pd.options.display.max_columns = 999

    dfbetaminus = pl.read_csv(
        at.get_config()["path_datadir"] / "betaminusdecays.txt",
        separator=" ",
        comment_prefix="#",
        has_header=False,
        new_columns=["A", "Z", "Q[MeV]", "E_gamma[MeV]", "E_elec[MeV]", "E_neutrino[MeV]", "tau[s]"],
    ).filter(pl.col("Q[MeV]") > 0.0)

    dfalpha = pl.read_csv(
        at.get_config()["path_datadir"] / "alphadecays.txt",
        separator=" ",
        comment_prefix="#",
        has_header=False,
        new_columns=[
            "A",
            "Z",
            "branch_alpha",
            "branch_beta",
            "halflife_s",
            "Q_total_alphadec[MeV]",
            "Q_total_betadec[MeV]",
            "E_alpha[MeV]",
            "E_gamma[MeV]",
            "E_beta[MeV]",
        ],
    )

    nuclist = sorted(list(dfbetaminus.select(["Z", "A"]).iter_rows()) + list(dfalpha.select(["Z", "A"]).iter_rows()))

    colreplacements = {
        "Rad Int.": "intensity",
        "Rad Ene.": "radiationenergy_kev",
        "Rad subtype": "radsubtype",
        "Par. Elevel": "parent_elevel",
    }

    for z, a in nuclist:
        strnuclide = elsymbols[z].lower() + str(a)
        print(f"\n(Z={z}) {strnuclide}")
        filename = f"{strnuclide}_lines.txt"
        outpath = outfolder / filename
        # if outpath.is_file():
        #     print(f"  {filename} already exists. skipping...")
        #     continue

        url = f"https://www.nndc.bnl.gov/nudat3/decaysearchdirect.jsp?nuc={strnuclide}&unc=standard&out=file"

        with requests.Session() as s:
            textdata = s.get(url).text
            textdata = textdata.replace("**********", "0.")
            # print(textdata)
            # match
            if "<pre>" not in textdata:
                print(f"  no table data returned from {url}")
                continue

            startindex = textdata.find("<pre>") + len("<pre>")
            endindex = textdata.rfind("</pre>")
            strtable = textdata[startindex:endindex].strip()
            strheader = strtable.strip().split("\n")[0].strip()
            # print(strheader)
            assert (
                strheader
                == "A  	Element	Z  	N  	Par. Elevel	Unc. 	JPi       	Dec Mode	T1/2 (txt)    	T1/2 (num)       "
                " 	Daughter	Radiation	Rad subtype 	Rad Ene.  	Unc       	EP Ene.   	Unc       	Rad Int.  	Unc      "
                " 	Dose        	Unc"
            )
            # dfnuclide = pd.read_fwf(io.StringIO(strtable))
            dfnuclide = pd.read_csv(io.StringIO(strtable), delimiter="\t", dtype={"Par. Elevel": str})

            newcols = []
            for colname in dfnuclide.columns:
                colname = colname.strip()
                if colname.startswith("Unc"):
                    colname = newcols[-1] + " (Unc)"
                if colname in colreplacements:
                    colname = colreplacements[colname]
                newcols.append(colname)
            dfnuclide.columns = newcols
            dfnuclide["Dec Mode"] = dfnuclide["Dec Mode"].str.strip()
            dfnuclide["Radiation"] = dfnuclide["Radiation"].str.strip()
            dfnuclide["radsubtype"] = dfnuclide["radsubtype"].str.strip()
            dfnuclide["parent_elevel"] = dfnuclide["parent_elevel"].str.strip()

            found_groundlevel = False
            for parelevel, dfdecay in dfnuclide.groupby("parent_elevel"):
                try:
                    is_groundlevel = float(parelevel) == 0.0
                except ValueError:
                    is_groundlevel = False
                print(f"  parent_Elevel: {parelevel} is_groundlevel: {is_groundlevel}")
                if not is_groundlevel:
                    continue
                found_groundlevel = True
                dfgammadecays = dfdecay.query(
                    "Radiation == 'G' and (radsubtype == '' or radsubtype == 'Annihil.') and intensity >= 0.15",
                    inplace=False,
                )

                maybedfbetaminusrow = dfbetaminus.filter(pl.col("Z") == z).filter(pl.col("A") == a)
                maybedfalpharow = dfalpha.filter(pl.col("Z") == z).filter(pl.col("A") == a)
                if not dfgammadecays.empty:
                    print(f"                     NNDC half-life: {dfgammadecays.iloc[0]['T1/2 (num)']:7.1e} s")

                if maybedfbetaminusrow.height > 0:
                    halflife = maybedfbetaminusrow["tau[s]"].item() * math.log(2)
                    print(f"      betaminusdecays.txt half-life: {halflife:7.1e} s")
                if maybedfalpharow.height > 0:
                    print(f"          alphadecays.txt half-life: {maybedfalpharow['halflife_s'].item():7.1e} s")

                e_gamma = (dfgammadecays["radiationenergy_kev"] * dfgammadecays["intensity"] / 100.0).sum()
                print(f"                   NNDC Egamma: {e_gamma:7.1f} keV")

                if maybedfbetaminusrow.height > 0:
                    file_e_gamma = maybedfbetaminusrow["E_gamma[MeV]"].item() * 1000
                    print(f"    betaminusdecays.txt Egamma: {file_e_gamma:7.1f} keV")
                    if not np.isclose(e_gamma, file_e_gamma, rtol=0.1):
                        print("WARNING!!!!!!")

                elif maybedfalpharow.height > 0:
                    file_e_gamma = maybedfalpharow["E_gamma[MeV]"].item() * 1000
                    print(f"        alphadecays.txt Egamma: {file_e_gamma:7.1f} keV")
                    if not np.isclose(e_gamma, file_e_gamma, rtol=0.1):
                        print("WARNING!!!!!!")

                dfout = pl.DataFrame(
                    {
                        "energy_mev": dfgammadecays.radiationenergy_kev.to_numpy() / 1000.0,
                        "intensity": dfgammadecays.intensity.to_numpy() / 100.0,
                    }
                ).sort("energy_mev")
                if len(dfout) > 0:
                    with outpath.open("w", encoding="utf-8") as fout:
                        fout.write(f"{len(dfout)}\n")
                        for energy_mev, intensity in dfout[["energy_mev", "intensity"]].iter_rows():
                            fout.write(f"{energy_mev:5.3f}  {intensity:6.4f}\n")

                        print(f"Saved {filename}")
                else:
                    print("empty DataFrame")
            if not found_groundlevel:
                print("  ERROR! did not find ground level")


if __name__ == "__main__":
    main()
