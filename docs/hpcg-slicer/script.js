document.addEventListener('DOMContentLoaded', function() {
    
    // --- Global DOM Elements and Data ---
    const instanceSlicer = document.getElementById('instance-slicer');
    const metricRadios = document.querySelectorAll('input[name="metric"]');
    const plot2dDiv = document.getElementById('plot2d');
    const plot3dDiv = document.getElementById('plot3d');
    let fullData = [];

    // --- Plotting Functions ---

    function updateHeatmap() {
        const selectedInstance = instanceSlicer.value;
        const selectedMetric = document.querySelector('input[name="metric"]:checked').value;
        const instanceData = fullData.filter(d => d.instance === selectedInstance);
        
        const xLabels = [...new Set(fullData.map(d => d.opt))].sort();
        const yLabels = [...new Set(fullData.map(d => d.arch))].sort();
        
        const zValues = yLabels.map(arch => 
            xLabels.map(opt => {
                const point = instanceData.find(d => d.arch === arch && d.opt === opt);
                return point ? point[selectedMetric] : null;
            })
        );
        
        const hoverText = yLabels.map(arch => 
            xLabels.map(opt => {
                const point = instanceData.find(d => d.arch === arch && d.opt === opt);
                if (!point) return 'No Data';
                return `Arch: ${point.arch}<br>Opt: ${point.opt}<br>` +
                       `Status: ${point.works ? 'NOMINAL' : 'FAILURE'}<br>FOM: ${point.fom}<br>` +
                       `CPU Time: ${point.cpu_time}`;
            })
        );

        const data = [{
            x: xLabels, y: yLabels, z: zValues, type: 'heatmap',
            hoverongaps: false, text: hoverText, hoverinfo: 'text',
            // Vibrant, space-themed color scales
            colorscale: selectedMetric === 'works' ? [[0, '#ff4136'], [1, '#00ff99']] : 'Plasma',
            showscale: selectedMetric !== 'works',
        }];

        const layout = {
            // Use Plotly's dark theme as a base
            template: 'plotly_dark',
            title: `Telemetry: <b>${selectedMetric.toUpperCase()}</b> // System: <b>${selectedInstance}</b>`,
            xaxis: { title: 'Optimization Level', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'Micro-architecture', gridcolor: 'rgba(255,255,255,0.1)' },
            paper_bgcolor: 'rgba(0,0,0,0)', // Transparent background
            plot_bgcolor: 'rgba(0,0,0,0)',
            autosize: true
        };
        
        Plotly.newPlot(plot2dDiv, data, layout, {responsive: true});
    }

    function update3DPlot() {
        const selectedInstance = instanceSlicer.value;
        const plotData = fullData.filter(d => d.works === true);

        const selectedTrace = {
            x: plotData.filter(d => d.instance === selectedInstance).map(d => d.opt),
            y: plotData.filter(d => d.instance === selectedInstance).map(d => d.arch),
            z: plotData.filter(d => d.instance === selectedInstance).map(d => d.instance),
            mode: 'markers', type: 'scatter3d', name: selectedInstance,
            marker: { 
                color: plotData.filter(d => d.instance === selectedInstance).map(d => d.fom), 
                colorscale: 'Plasma', // Vibrant colors for highlighted points
                size: 12, 
                showscale: true, 
                colorbar: {title: 'FOM', tickfont: {color: '#d0d0d0'}, titlefont: {color: '#d0d0d0'}} 
            },
            text: plotData.filter(d => d.instance === selectedInstance).map(d => `System: ${d.instance}<br><b>FOM: ${d.fom}</b>`),
            hoverinfo: 'text+name'
        };

        const otherTrace = {
            x: plotData.filter(d => d.instance !== selectedInstance).map(d => d.opt),
            y: plotData.filter(d => d.instance !== selectedInstance).map(d => d.arch),
            z: plotData.filter(d => d.instance !== selectedInstance).map(d => d.instance),
            mode: 'markers', type: 'scatter3d', name: 'Other Systems',
            // Faint cyan for non-selected points, like faint stars
            marker: { color: 'rgba(0, 198, 255, 0.5)', size: 6 },
            text: plotData.filter(d => d.instance !== selectedInstance).map(d => `System: ${d.instance}<br>FOM: ${d.fom}`),
            hoverinfo: 'text+name'
        };

        const layout = {
            template: 'plotly_dark',
            title: 'HPC Performance Constellation',
            margin: { l: 0, r: 0, b: 0, t: 40 },
            autosize: true,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            scene: {
                xaxis: { title: 'Optimization', gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: 'Architecture', gridcolor: 'rgba(255,255,255,0.1)' },
                zaxis: { title: 'System Type', gridcolor: 'rgba(255,255,255,0.1)' }
            },
            legend: { x: 0.7, y: 0.95, font: {color: '#d0d0d0'} }
        };

        Plotly.newPlot(plot3dDiv, [otherTrace, selectedTrace], layout, {responsive: true});
    }

    // --- Data Loading and Initialization (Unchanged) ---
    fetch('data.json')
        .then(response => response.json())
        .then(data => {
            fullData = data;
            
            const instances = [...new Set(fullData.map(d => d.instance))].sort();
            instances.forEach(instance => {
                const option = document.createElement('option');
                option.value = option.textContent = instance;
                instanceSlicer.appendChild(option);
            });

            instanceSlicer.addEventListener('change', () => {
                updateHeatmap();
                update3DPlot();
            });
            metricRadios.forEach(radio => radio.addEventListener('change', updateHeatmap));

            updateHeatmap();
            update3DPlot();
        });
});
