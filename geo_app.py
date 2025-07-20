import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import io
import zipfile
from scipy.io import savemat
from PIL import Image  # PNG to TIFF conversion

# ---------------------------------------------
# Geodezik Kubbe Fonksiyonları
# ---------------------------------------------
POLYHEDRON_NAMES = {
    3: 'Tetrahedron',
    4: 'Octahedron',
    5: 'Icosahedron',
    6: 'Hexahedron',
    7: 'Heptahedron',
    8: 'Octahedron_8',
    9: 'Enneahedron'
}

def generate_dome_geometry(params):
    t = params.get('type')
    if t not in range(3, 10):
        raise ValueError('Invalid face number. Must be between 3 and 9.')
    face = t
    dome = params.copy()
    dome['poly_name'] = POLYHEDRON_NAMES.get(face, f'Polyhedron_{face}')
    return polyhedron_base(dome, face)

def polyhedron_base(dome, face):
    sp = dome['span']
    ht = dome['height']
    fr = dome['freq']
    R = ((sp / 2) ** 2 / ht + ht) / 2
    if ht > R:
        raise ValueError('Dome height cannot exceed radius.')
    alpha = np.degrees(np.arcsin(sp / (2 * R)))
    phi = alpha / fr
    dome['radius'], dome['angle'], dome['fr_angle'] = R, alpha, phi

    # Düğüm sayısı
    ds = sum(range(1, fr + 1)) * face + 1
    dome['nodenum'] = ds
    nodes = np.zeros((ds, 4))

    # Eleman sayısı
    es = (sum(range(1, fr + 1)) * 3 - fr) * face
    dome['memnum'] = es
    eleman = np.zeros((es, 2), dtype=int)
    mem = np.zeros((es, 5))

    # Tepe noktası
    nodes[0] = [1, 0, 0, ht]
    dx = [nodes[0:1]]

    # Her frekans katmanı için düğüm noktalarını oluştur
    j = 0
    for i in range(1, fr + 1):
        rx = R * np.sin(np.radians(phi * i))
        ar = 2 * np.pi / (face * i)
        aci = np.arange(0, 2 * np.pi, ar)
        ll = np.arange(j + 1, j + 1 + len(aci))

        nodes[ll, 0] = ll + 1
        nodes[ll, 1] = rx * np.cos(aci)
        nodes[ll, 2] = rx * np.sin(aci)
        # Kubbenin yüksekliği R - (R - ht) kadar ötelenmiş
        nodes[ll, 3] = R * np.cos(np.radians(phi * i)) - (R - ht)

        j += len(aci)
        dx.append(nodes[ll])

    dome['nodes'] = nodes

    # Eleman (member) tanımlaması
    idx = 0
    top_idx = int(dx[0][0, 0]) - 1           # Tepe düğümü indisi (0 tabanlı)
    ring_idx = (dx[1][:, 0] - 1).astype(int)  # Birinci halka düğüm indisleri

    # Tepe ile birinci halkayı birbirine bağlayan dikey elemanlar
    for nid in ring_idx:
        eleman[idx] = [nid, top_idx]
        idx += 1

    # Merkezden aşağıya doğru köşegen elemanlar
    for i in range(1, fr):
        d2 = (dx[i + 1][::i + 1, 0] - 1).astype(int)
        d1 = (dx[i][::i, 0] - 1).astype(int)
        for a, b in zip(d1, d2):
            eleman[idx] = [a, b]
            idx += 1

    # Yatay (çevresel) elemanlar
    for i in range(1, fr + 1):
        k = (dx[i][:, 0] - 1).astype(int)
        for a, b in zip(k, np.roll(k, -1)):
            eleman[idx] = [a, b]
            idx += 1

    # Her alt bölmede iki köşegen eleman
    for i in range(1, fr):
        d2 = (dx[i + 1][:, 0] - 1).astype(int)
        d1 = (dx[i][:, 0] - 1).astype(int)
        d2x = (dx[i + 1][::i + 1, 0] - 1).astype(int)
        tmp = np.hstack([np.setdiff1d(d2, d2x)[-1:], np.setdiff1d(d2, d2x)])
        for a, t1, t2 in zip(d1, tmp, np.roll(tmp, -1)):
            eleman[idx] = [a, t1]
            eleman[idx + 1] = [a, t2]
            idx += 2

    mem[:, 0] = np.arange(1, len(eleman) + 1)
    mem[:, 1:3] = eleman + 1   # 1-bazlı düğüm indeksleri
    dome['members'] = mem

    # Destek (supports)
    supp = dx[-1].copy()
    supp[:, 1:4] = 1
    dome['support'] = supp.astype(int)

    return dome

def grpdet(dome, tol=1e-6):
    pairs = dome['members'][:, 1:3].astype(int) - 1  # 0 bazlı
    lengths = np.linalg.norm(
        dome['nodes'][pairs[:, 0], 1:4] - dome['nodes'][pairs[:, 1], 1:4],
        axis=1
    )
    groups, unique = {}, []
    for idx, L in enumerate(lengths):
        for gid, UL in enumerate(unique):
            if abs(L - UL) < tol:
                groups.setdefault(gid, []).append(idx)
                break
        else:
            unique.append(L)
            groups[len(unique) - 1] = [idx]
    dome['groups'] = groups
    dome['lengths'] = lengths
    dome['group_count'] = len(groups)
    dome['total_length'] = float(np.sum(lengths))
    return dome

def define_supports(dome):
    dome['supports'] = dome['support'][:, 0].astype(int)
    return dome

def save_dome_info(dome, filename_prefix):
    info = {
        'polyhedron': dome['poly_name'],
        'frequency': int(dome['freq']),
        'node_count': int(dome['nodenum']),
        'element_count': int(dome['memnum']),
        'group_count': int(dome['group_count']),
        'total_length': dome['total_length']
    }
    df_info = pd.DataFrame([info])
    df_nodes = pd.DataFrame(dome['nodes'], columns=['Node', 'X', 'Y', 'Z']).astype({'Node': int})
    df_members = pd.DataFrame({
        'Member': dome['members'][:, 0].astype(int),
        'Node1': dome['members'][:, 1].astype(int),
        'Node2': dome['members'][:, 2].astype(int)
    })
    group_list = []
    for gid, elems in dome['groups'].items():
        length_val = dome['lengths'][elems[0]]
        group_list.append({'Group': gid + 1, 'Length': round(length_val, 4), 'Elements': elems})
    df_groups = pd.DataFrame(group_list)
    df_supports = pd.DataFrame(dome['supports'], columns=['SupportNode']).astype({'SupportNode': int})
    csv_buffers = {
        'info.csv': df_info.to_csv(index=False).encode('utf-8'),
        'nodes.csv': df_nodes.to_csv(index=False).encode('utf-8'),
        'members.csv': df_members.to_csv(index=False).encode('utf-8'),
        'groups.csv': df_groups.to_csv(index=False).encode('utf-8'),
        'supports.csv': df_supports.to_csv(index=False).encode('utf-8')
    }
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df_info.to_excel(writer, sheet_name='Info', index=False)
        df_nodes.to_excel(writer, sheet_name='Nodes', index=False)
        df_members.to_excel(writer, sheet_name='Members', index=False)
        df_groups.to_excel(writer, sheet_name='Groups', index=False)
        df_supports.to_excel(writer, sheet_name='Supports', index=False)
    excel_data = excel_buffer.getvalue()
    mat_dict = {
        'info': info,
        'nodes': dome['nodes'],
        'members': dome['members'],
        'groups': dome['groups'],
        'lengths': dome['lengths'],
        'supports': dome['supports']
    }
    mat_buffer = io.BytesIO()
    savemat(mat_buffer, mat_dict)
    mat_data = mat_buffer.getvalue()
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode='w') as zf:
        for name, data in csv_buffers.items():
            zf.writestr(name, data)
        zf.writestr('dome_info.xlsx', excel_data)
        zf.writestr('dome_data.mat', mat_data)
    zip_buffer.seek(0)
    return zip_buffer

st.set_page_config(layout="wide", page_title="Geodesic Dome Web App")
st.title("Geodezik Dome Geometry App")

type_face = st.sidebar.selectbox(
    "Dome Type (face number)",
    options=list(POLYHEDRON_NAMES.keys()),
    format_func=lambda x: f"{POLYHEDRON_NAMES[x]}-{x}"
)
span = st.sidebar.number_input("Span", min_value=1.0, value=31.78)
height = st.sidebar.number_input("Height", min_value=0.1, value=7.0)
freq = st.sidebar.number_input("Frequency", min_value=1, value=5)
st.sidebar.markdown("### Display Options")
show_nodes = st.sidebar.checkbox("Show Nodes", value=True)
show_members = st.sidebar.checkbox("Show Members", value=True)
show_groups = st.sidebar.checkbox("Show Groups", value=False)
if show_groups and show_members:
    show_members = False

group_style = st.sidebar.radio(
    "Group Color Style",
    options=["Single Color", "Colored"]
)
if st.sidebar.button("Generate Dome"):
    params = {'type': type_face, 'span': span, 'height': height, 'freq': freq}
    dome = generate_dome_geometry(params)
    dome = grpdet(dome)
    dome = define_supports(dome)
    st.session_state.update({
        'dome': dome,
        'show_nodes': show_nodes,
        'show_members': show_members,
        'show_groups': show_groups,
        'group_style': group_style
    })

if 'dome' in st.session_state:
    dome = st.session_state['dome']
    show_nodes = st.session_state['show_nodes']
    show_members = st.session_state['show_members']
    show_groups = st.session_state['show_groups']
    group_style = st.session_state['group_style']

    st.subheader("Dome Information")
    lengths = dome['lengths']
    min_len = round(float(lengths.min()), 4)
    max_len = round(float(lengths.max()), 4)
    info_dict = {
        'Dome Type': dome['poly_name'],
        'Frequency': dome['freq'],
        'Node Count': dome['nodenum'],
        'Member Count': dome['memnum'],
        'Group Count': dome['group_count'],
        'Total Length': round(dome['total_length'], 4),
        'Min Element Length': min_len,
        'Max Element Length': max_len
    }
    df_info_vert = pd.DataFrame.from_dict(info_dict, orient='index', columns=['Value'])
    df_info_vert['Value'] = df_info_vert['Value'].astype(str)
    st.table(df_info_vert)

    st.subheader("Nodes")
    df_nodes = pd.DataFrame(dome['nodes'], columns=['Node', 'X', 'Y', 'Z']).astype({'Node': int})
    st.dataframe(df_nodes)

    st.subheader("Members")
    df_members = pd.DataFrame({
        'Member': dome['members'][:, 0].astype(int),
        'Node1': dome['members'][:, 1].astype(int),
        'Node2': dome['members'][:, 2].astype(int)
    })
    st.dataframe(df_members)

    st.subheader("Dome Visualization")
    fig = go.Figure()
    nodes = dome['nodes']
    members = dome['members'][:, 1:3].astype(int)

        # Elemanları çiz: Eğer Colored mod ise her grup renkli, değilse gri
    if group_style == 'Colored':
        palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', 
                   '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', 
                   '#FF97FF', '#FECB52']
        group_mx, group_my, group_mz, group_text, text_color_list = [], [], [], [], []
        for gid, elems in dome['groups'].items():
            color = palette[gid % len(palette)]
            for ei in elems:
                a, b = members[ei] - 1
                p1, p2 = nodes[a, 1:4], nodes[b, 1:4]
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                    mode='lines', line=dict(color=color, width=3), showlegend=False
                ))
                # Grup etiketleri için midpoint ekle
                mx, my, mz = (p1 + p2) / 2
                group_mx.append(mx)
                group_my.append(my)
                group_mz.append(mz)
                group_text.append(str(gid + 1))
                text_color_list.append(color)
        # Grup metinlerini çiz
        fig.add_trace(go.Scatter3d(
            x=group_mx, y=group_my, z=group_mz,
            mode='text', text=group_text,
            textposition='middle center',
            textfont=dict(size=8, color=text_color_list),
            name='Groups'
        ))
    else:
        for a, b in members:
            p1, p2 = nodes[a-1, 1:4], nodes[b-1, 1:4]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode='lines', line=dict(color='lightgray', width=2), showlegend=False
            ))

    if show_nodes:
            fig.add_trace(go.Scatter3d(
            x=nodes[:, 1], y=nodes[:, 2], z=nodes[:, 3],
            mode='markers+text',
            marker=dict(size=4, color='black'),
            text=nodes[:, 0].astype(int),
            textposition='top center',
            textfont=dict(size=8, color='black'),
            name='Nodes'
        ))

    if show_members:
        elem_mx, elem_my, elem_mz, elem_text = [], [], [], []
        for idx, (a, b) in enumerate(members, start=1):
            p1, p2 = nodes[a-1, 1:4], nodes[b-1, 1:4]
            mx, my, mz = (p1 + p2) / 2
            elem_mx.append(mx)
            elem_my.append(my)
            elem_mz.append(mz)
            elem_text.append(str(idx))
        fig.add_trace(go.Scatter3d(
            x=elem_mx, y=elem_my, z=elem_mz,
            mode='text',
            text=elem_text,
            textposition='middle center',
            textfont=dict(size=8, color='black'),
            name='Members'
        ))

    x_vals, y_vals, z_vals = nodes[:, 1], nodes[:, 2], nodes[:, 3]
    x_range = x_vals.max() - x_vals.min()
    y_range = y_vals.max() - y_vals.min()
    z_range = z_vals.max() - z_vals.min()
    center = [x_vals.mean(), y_vals.mean(), z_vals.mean()]
    camera = dict(eye=dict(x=1.5 * x_range + center[0],
                            y=1.5 * y_range + center[1],
                            z=1.5 * z_range + center[2]))
    fig.update_layout(
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   aspectmode='manual',
                   aspectratio=dict(x=x_range, y=y_range, z=z_range),
                   camera=camera),
        legend=dict(y=0.1),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # TIFF Download
    png_bytes = fig.to_image(format="png", scale=4)
    img = Image.open(io.BytesIO(png_bytes))
    tiff_buffer = io.BytesIO()
    img.save(tiff_buffer, format="TIFF", dpi=(300,300))
    tiff_bytes = tiff_buffer.getvalue()
    st.download_button(
        label="Download Figure as TIFF",
        data=tiff_bytes,
        file_name="dome_figure.tiff",
        mime="image/tiff"
    )

    # PDF Download
    pdf_bytes = fig.to_image(format="pdf")
    st.download_button(
        label="Download Figure as PDF",
        data=pdf_bytes,
        file_name="dome_figure.pdf",
        mime="application/pdf"
    )

    # ZIP indirme
    st.subheader("Download Results")
    zip_data = save_dome_info(dome, filename_prefix='dome_output')
    st.download_button(
        label="Download All Results (ZIP)",
        data=zip_data,
        file_name="dome_results.zip",
        mime="application/zip"
    )
