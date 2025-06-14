document.addEventListener('DOMContentLoaded', function () {
    const selectElement = document.getElementById('instanceTypeSelect');
    const heatmapContainer = document.getElementById('heatmapContainer');
    const platformBox = document.getElementById('platformBox'); 
    let allData = null;
    function renderPlotlyHeatmap(instanceName) {
         if (!allData || !allData.instances[instanceName]) {
             Plotly.purge(heatmapContainer); // Clear existing plot
             heatmapContainer.innerHTML = "<p>No data available for this instance.</p>";
             return;
         }
         const details = allData.instances[instanceName];
         const colorCompatible = '#00C0FF'; // Yes
         const colorIncompatible = '#FF4136'; // No
         const pageBackgroundColor = '#0c0c1e';
         const trace = {
             z: details.heatmap,
             x: details.microarches,
             y: details.optimizations,
             type: 'heatmap',
             colorscale: [ // This ensures discrete mapping for 0 and 1
                [0, colorIncompatible],
                [1, colorCompatible]
             ],
             showscale: false,
             colorbar: {
                 title: 'Compatibility',
                 tickvals: [0, 1],
                 ticktext: ['No', 'Yes']
             },
             zmin: 0,
             zmax: 1,
             hoverongaps: false,
             hovertemplate: 'Micro-arch: %{x}<br>Optimization: %{y}<br>Compatible: %{z}<extra></extra>'
         };
         if (details.platform) {
            platformBox.textContent = `${details.platform}`;
         } else {
            platformBox.textContent = "Platform N/A";
         }

      const layout = {
        template: "plotly_dark",
        title: {
            text: `Compatibility Scan: ${instanceName}`,
            font: { color: '#ffffff', size: 20 },
            x: 0.5,
            xanchor: 'center'
        },
        xaxis: {
            title: 'Target Micro-architecture Array',
            side: 'top',
            automargin: true,
            tickangle: -30,
            gridcolor: 'rgba(100,100,150,0.2)',
            linecolor: '#5070b0', // Still useful for axis lines
            tickfont: { color: '#a0a0ff', size: 10 },
            titlefont: { color: '#c0c0ff', size: 12 }
        },
        yaxis: {
            title: 'Optimization Strategy',
            autorange: 'reversed',
            automargin: true,
            gridcolor: 'rgba(100,100,150,0.2)',
            linecolor: '#5070b0', // Still useful for axis lines
            tickfont: { color: '#a0a0ff', size: 10 },
            titlefont: { color: '#c0c0ff', size: 12 }
        },
        paper_bgcolor: pageBackgroundColor,
        plot_bgcolor: pageBackgroundColor,
        margin: { t: 140 }
    };
         Plotly.react(heatmapContainer, [trace], layout); // Use react for efficient updates
     }

    // --- Load data and initialize ---
    // Using fetch for compatibility_data.json
    fetch('compatibility_data.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            allData = data;
            // Populate dropdown with ordered instance types
            allData.instance_names.forEach(instanceName => {
                const option = document.createElement('option');
                option.value = instanceName;
                option.textContent = instanceName;
                selectElement.appendChild(option);
            });

            // Add event listener
            selectElement.addEventListener('change', function () {
                renderPlotlyHeatmap(this.value);
            });

            // Initial render for the first selected instance
            if (allData.instance_names.length > 0) {
                selectElement.value = allData.instance_names[0];
                renderPlotlyHeatmap(allData.instance_names[0]);
            } else {
                 heatmapContainer.innerHTML = "<p>No instance types found in data.</p>";
                 platformBox.textContent = "--";
            }
        })
        .catch(error => {
            console.error('Error loading or processing data:', error);
            heatmapContainer.innerHTML = `<p>Error loading data: ${error.message}. Check console.</p>`;
        });
});
