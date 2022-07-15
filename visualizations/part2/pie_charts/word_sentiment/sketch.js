
// Set the dimensions and margins of the graph
const SCREEN_DIMENSIONS = { 
    width: 850,
    height: 400,
    leftMargin: 10,
    rightMargin: 115,
    topMargin: 10,
    bottomMargin: 100
};
SCREEN_DIMENSIONS.innerWidth = SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.leftMargin - SCREEN_DIMENSIONS.rightMargin;
SCREEN_DIMENSIONS.innerHeight = SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.topMargin - SCREEN_DIMENSIONS.bottomMargin;


// Run `cleanDataOutput` in `file_manager.py` to ensure most recent data
const DATA_LOC = '../../data/words_sentiment_distrib.csv';
// const DATA_LOC = 'https://raw.githubusercontent.com/peterg7/Russia_Ukraine_Twitter_Sentiment_Analysis/main/visualizations/part1/data/words_sentiment_distrib.csv';
const TITLE = "Sentiment Distribution of Words (Generalized)";

// Append the svg object to the body of the page
var svg = d3.select('body').append('svg')
            .attr('width', SCREEN_DIMENSIONS.width)
            .attr('height', SCREEN_DIMENSIONS.height)
            .attr("viewBox", [-SCREEN_DIMENSIONS.leftMargin, -SCREEN_DIMENSIONS.topMargin, 
                                SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");


const colorParams = {
    value: row => row.data.sentiment
} 

const labelParams = {
    value: row => row.value
} 

const canvas = svg.append('g')
    .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin/2},${SCREEN_DIMENSIONS.topMargin/2})`);

const titleGroup = canvas.append("text")
    .attr('transform', `translate(${(SCREEN_DIMENSIONS.innerWidth / 2)},${0 - (SCREEN_DIMENSIONS.topMargin)})`)
    .attr("class", "title")
    .text(TITLE);

const legendGroup = canvas.append('g')
    .attr("transform", `translate(${SCREEN_DIMENSIONS.innerWidth - SCREEN_DIMENSIONS.rightMargin},${SCREEN_DIMENSIONS.topMargin * 2})`)
    .attr("class", "legendOrdinal");


const g = svg.append('g')
    .attr('transform', `translate(${SCREEN_DIMENSIONS.width/2.3},${SCREEN_DIMENSIONS.height/1.95})`);

var pie = d3.pie().value(d => d.percent);

var radius = Math.min(SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height) / 2;

var path = d3.arc()
             .outerRadius(radius - 10)
             .innerRadius(0);

var label = d3.arc()
              .outerRadius(radius)
              .innerRadius(radius - 120);



const preprocess = (row, i) => {
    // Know there are no null-values from preprocessing
    return {
        'sentiment': row['sentiment'],
        'count': parseInt(row['count']),
        'percent': parseFloat(row['percent'])
    }
};

const percentRanges = arr => {
    var scaled = []
    let carryover = 0
    for (let i = 0; i < arr.length; ++i){
        scaled.push({
            'sentiment': arr[i].sentiment, 
            'percent': arr[i].percent,
            'range': [carryover, arr[i].percent + carryover]
        })
        carryover += arr[i].percent
    }
    return scaled
}


d3.csv(DATA_LOC, preprocess).then((data, i) => {


    var colorScale = d3.scaleOrdinal()
                    .domain(['negative', 'positive', 'neutral'])
                    .range(['#e41a1c', '#4daf4a', '#377eb8'])


    var arc = g.selectAll(".arc")
                        .data(pie(data))
                        .enter().append("g")
                        .attr("class", "arc");

    arc.append("path")
        .attr("d", path)
        .attr("fill", d => colorScale(colorParams.value(d)));
    
    arc.append("text")
        .attr("transform", d => "translate(" + label.centroid(d) + ")")
        .text(d => d3.format(".1%")(labelParams.value(d) / 100));


    // Define and build legend
    var legend = d3.legendColor()
        .shapeHeight(15)
        .orient("vertical")
        .scale(colorScale) 
        .title("Sentiment")
        .titleWidth(30);

    legendGroup.call(legend);
});