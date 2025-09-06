"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var react_1 = require("react");
var recharts_1 = require("recharts");
var lucide_react_1 = require("lucide-react");
var MortgageModelMonitoringDashboard = function () {
    var _a = (0, react_1.useState)('7d'), selectedTimeRange = _a[0], setSelectedTimeRange = _a[1];
    var _b = (0, react_1.useState)('accuracy'), selectedMetric = _b[0], setSelectedMetric = _b[1];
    var _c = (0, react_1.useState)(300), refreshInterval = _c[0], setRefreshInterval = _c[1]; // 5 minutes
    var _d = (0, react_1.useState)(new Date()), lastUpdated = _d[0], setLastUpdated = _d[1];
    // Simulated real-time data - in production, this would come from CloudWatch/SageMaker Model Monitor
    var _e = (0, react_1.useState)({
        modelPerformance: [
            { date: '2024-01-01', accuracy: 0.923, precision: 0.891, recall: 0.876, f1Score: 0.883, auc: 0.945 },
            { date: '2024-01-02', accuracy: 0.918, precision: 0.885, recall: 0.872, f1Score: 0.878, auc: 0.941 },
            { date: '2024-01-03', accuracy: 0.925, precision: 0.894, recall: 0.881, f1Score: 0.887, auc: 0.948 },
            { date: '2024-01-04', accuracy: 0.921, precision: 0.889, recall: 0.874, f1Score: 0.881, auc: 0.943 },
            { date: '2024-01-05', accuracy: 0.915, precision: 0.882, recall: 0.868, f1Score: 0.875, auc: 0.938 },
            { date: '2024-01-06', accuracy: 0.912, precision: 0.878, recall: 0.865, f1Score: 0.871, auc: 0.935 },
            { date: '2024-01-07', accuracy: 0.908, precision: 0.874, recall: 0.861, f1Score: 0.867, auc: 0.931 }
        ],
        dataDrift: [
            { feature: 'debt_to_income', baseline: 0.35, current: 0.38, drift: 0.086, status: 'warning' },
            { feature: 'loan_to_value', baseline: 0.75, current: 0.77, drift: 0.027, status: 'normal' },
            { feature: 'credit_score', baseline: 720, current: 715, drift: 0.042, status: 'normal' },
            { feature: 'income', baseline: 85000, current: 87500, drift: 0.029, status: 'normal' },
            { feature: 'employment_length', baseline: 8.5, current: 8.2, drift: 0.035, status: 'normal' },
            { feature: 'property_value', baseline: 425000, current: 440000, drift: 0.035, status: 'normal' }
        ],
        predictionDistribution: [
            { range: '0.0-0.1', count: 1250, percentage: 41.7 },
            { range: '0.1-0.2', count: 780, percentage: 26.0 },
            { range: '0.2-0.3', count: 450, percentage: 15.0 },
            { range: '0.3-0.4', count: 280, percentage: 9.3 },
            { range: '0.4-0.5', count: 150, percentage: 5.0 },
            { range: '0.5+', count: 90, percentage: 3.0 }
        ],
        businessMetrics: [
            { date: '2024-01-01', totalPredictions: 2890, avgProcessingTime: 125, errorRate: 0.002, throughput: 48.2 },
            { date: '2024-01-02', totalPredictions: 3120, avgProcessingTime: 128, errorRate: 0.001, throughput: 52.0 },
            { date: '2024-01-03', totalPredictions: 2950, avgProcessingTime: 122, errorRate: 0.003, throughput: 49.2 },
            { date: '2024-01-04', totalPredictions: 3200, avgProcessingTime: 130, errorRate: 0.002, throughput: 53.3 },
            { date: '2024-01-05', totalPredictions: 3050, avgProcessingTime: 127, errorRate: 0.001, throughput: 50.8 },
            { date: '2024-01-06', totalPredictions: 2980, avgProcessingTime: 124, errorRate: 0.002, throughput: 49.7 },
            { date: '2024-01-07', totalPredictions: 3100, avgProcessingTime: 129, errorRate: 0.003, throughput: 51.7 }
        ],
        alerts: [
            { id: 1, type: 'warning', message: 'Data drift detected in debt_to_income feature', timestamp: '2024-01-07 14:30', severity: 'medium' },
            { id: 2, type: 'info', message: 'Model performance within acceptable range', timestamp: '2024-01-07 12:00', severity: 'low' },
            { id: 3, type: 'success', message: 'Batch inference job completed successfully', timestamp: '2024-01-07 10:15', severity: 'low' }
        ]
    }), dashboardData = _e[0], setDashboardData = _e[1];
    // Simulate real-time updates
    (0, react_1.useEffect)(function () {
        var interval = setInterval(function () {
            setLastUpdated(new Date());
            // In production, this would fetch real data from AWS CloudWatch/SageMaker
        }, refreshInterval * 1000);
        return function () { return clearInterval(interval); };
    }, [refreshInterval]);
    var getStatusIcon = function (status) {
        switch (status) {
            case 'normal': return <lucide_react_1.CheckCircle className="w-4 h-4 text-green-500"/>;
            case 'warning': return <lucide_react_1.AlertTriangle className="w-4 h-4 text-yellow-500"/>;
            case 'critical': return <lucide_react_1.XCircle className="w-4 h-4 text-red-500"/>;
            default: return <lucide_react_1.AlertCircle className="w-4 h-4 text-gray-500"/>;
        }
    };
    var getAlertIcon = function (type) {
        switch (type) {
            case 'success': return <lucide_react_1.CheckCircle className="w-4 h-4 text-green-500"/>;
            case 'warning': return <lucide_react_1.AlertTriangle className="w-4 h-4 text-yellow-500"/>;
            case 'error': return <lucide_react_1.XCircle className="w-4 h-4 text-red-500"/>;
            default: return <lucide_react_1.AlertCircle className="w-4 h-4 text-blue-500"/>;
        }
    };
    var currentMetrics = dashboardData.modelPerformance[dashboardData.modelPerformance.length - 1];
    var previousMetrics = dashboardData.modelPerformance[dashboardData.modelPerformance.length - 2];
    var calculateTrend = function (current, previous) {
        var change = ((current - previous) / previous) * 100;
        return { change: change.toFixed(2), trend: change >= 0 ? 'up' : 'down' };
    };
    var COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];
    return (<div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-3xl font-bold text-gray-900">Mortgage ML Model Monitoring</h1>
            <div className="flex items-center space-x-4">
              <select value={selectedTimeRange} onChange={function (e) { return setSelectedTimeRange(e.target.value); }} className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value="1d">Last 24 Hours</option>
                <option value="7d">Last 7 Days</option>
                <option value="30d">Last 30 Days</option>
              </select>
              <select value={refreshInterval} onChange={function (e) { return setRefreshInterval(Number(e.target.value)); }} className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
                <option value={60}>1 minute</option>
                <option value={300}>5 minutes</option>
                <option value={900}>15 minutes</option>
              </select>
            </div>
          </div>
          <div className="flex items-center text-sm text-gray-500">
            <lucide_react_1.Clock className="w-4 h-4 mr-1"/>
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            {
                label: 'Model Accuracy',
                value: (currentMetrics.accuracy * 100).toFixed(1) + '%',
                trend: calculateTrend(currentMetrics.accuracy, previousMetrics.accuracy),
                icon: <lucide_react_1.Activity className="w-6 h-6 text-blue-500"/>
            },
            {
                label: 'Precision',
                value: (currentMetrics.precision * 100).toFixed(1) + '%',
                trend: calculateTrend(currentMetrics.precision, previousMetrics.precision),
                icon: <lucide_react_1.TrendingUp className="w-6 h-6 text-green-500"/>
            },
            {
                label: 'Daily Predictions',
                value: dashboardData.businessMetrics[dashboardData.businessMetrics.length - 1].totalPredictions.toLocaleString(),
                trend: { change: '+5.2', trend: 'up' },
                icon: <lucide_react_1.Eye className="w-6 h-6 text-purple-500"/>
            },
            {
                label: 'Error Rate',
                value: (dashboardData.businessMetrics[dashboardData.businessMetrics.length - 1].errorRate * 100).toFixed(3) + '%',
                trend: { change: '-12.5', trend: 'down' },
                icon: <lucide_react_1.Database className="w-6 h-6 text-orange-500"/>
            }
        ].map(function (metric, index) { return (<div key={index} className="bg-white rounded-lg shadow-sm border p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{metric.label}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{metric.value}</p>
                </div>
                {metric.icon}
              </div>
              <div className={"flex items-center mt-4 text-sm ".concat(metric.trend.trend === 'up' ? 'text-green-600' : 'text-red-600')}>
                {metric.trend.trend === 'up' ?
                <lucide_react_1.TrendingUp className="w-4 h-4 mr-1"/> :
                <lucide_react_1.TrendingDown className="w-4 h-4 mr-1"/>}
                {metric.trend.change}% from yesterday
              </div>
            </div>); })}
        </div>

        {/* Model Performance Trends */}
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-8">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-900">Model Performance Trends</h2>
            <select value={selectedMetric} onChange={function (e) { return setSelectedMetric(e.target.value); }} className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500">
              <option value="accuracy">Accuracy</option>
              <option value="precision">Precision</option>
              <option value="recall">Recall</option>
              <option value="f1Score">F1 Score</option>
              <option value="auc">AUC</option>
            </select>
          </div>
          <recharts_1.ResponsiveContainer width="100%" height={300}>
            <recharts_1.LineChart data={dashboardData.modelPerformance}>
              <recharts_1.CartesianGrid strokeDasharray="3 3"/>
              <recharts_1.XAxis dataKey="date"/>
              <recharts_1.YAxis domain={['dataMin - 0.01', 'dataMax + 0.01']}/>
              <recharts_1.Tooltip formatter={function (value) { return [(value * 100).toFixed(1) + '%', selectedMetric.toUpperCase()]; }}/>
              <recharts_1.Legend />
              <recharts_1.Line type="monotone" dataKey={selectedMetric} stroke="#2563eb" strokeWidth={2} dot={{ fill: '#2563eb', strokeWidth: 2, r: 4 }}/>
            </recharts_1.LineChart>
          </recharts_1.ResponsiveContainer>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Data Drift Monitoring */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Data Drift Detection</h2>
            <div className="space-y-4">
              {dashboardData.dataDrift.map(function (feature, index) { return (<div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(feature.status)}
                    <div>
                      <p className="font-medium text-gray-900">{feature.feature}</p>
                      <p className="text-sm text-gray-500">
                        Baseline: {typeof feature.baseline === 'number' ? feature.baseline.toLocaleString() : feature.baseline}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-gray-900">
                      {typeof feature.current === 'number' ? feature.current.toLocaleString() : feature.current}
                    </p>
                    <p className={"text-sm ".concat(feature.drift > 0.05 ? 'text-red-600' :
                feature.drift > 0.03 ? 'text-yellow-600' : 'text-green-600')}>
                      {(feature.drift * 100).toFixed(1)}% drift
                    </p>
                  </div>
                </div>); })}
            </div>
          </div>

          {/* Prediction Distribution */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Risk Score Distribution</h2>
            <recharts_1.ResponsiveContainer width="100%" height={250}>
              <recharts_1.BarChart data={dashboardData.predictionDistribution}>
                <recharts_1.CartesianGrid strokeDasharray="3 3"/>
                <recharts_1.XAxis dataKey="range"/>
                <recharts_1.YAxis />
                <recharts_1.Tooltip formatter={function (value, name) { return [value.toLocaleString(), 'Predictions']; }}/>
                <recharts_1.Bar dataKey="count" fill="#3b82f6"/>
              </recharts_1.BarChart>
            </recharts_1.ResponsiveContainer>
          </div>
        </div>

        {/* Business Metrics and System Performance */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Processing Time and Throughput */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">System Performance</h2>
            <recharts_1.ResponsiveContainer width="100%" height={250}>
              <recharts_1.LineChart data={dashboardData.businessMetrics}>
                <recharts_1.CartesianGrid strokeDasharray="3 3"/>
                <recharts_1.XAxis dataKey="date"/>
                <recharts_1.YAxis yAxisId="left"/>
                <recharts_1.YAxis yAxisId="right" orientation="right"/>
                <recharts_1.Tooltip />
                <recharts_1.Legend />
                <recharts_1.Line yAxisId="left" type="monotone" dataKey="avgProcessingTime" stroke="#8884d8" name="Avg Processing Time (ms)"/>
                <recharts_1.Line yAxisId="right" type="monotone" dataKey="throughput" stroke="#82ca9d" name="Throughput (req/min)"/>
              </recharts_1.LineChart>
            </recharts_1.ResponsiveContainer>
          </div>

          {/* Alert Summary */}
          <div className="bg-white rounded-lg shadow-sm border p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Recent Alerts</h2>
            <div className="space-y-4">
              {dashboardData.alerts.map(function (alert) { return (<div key={alert.id} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{alert.message}</p>
                    <p className="text-sm text-gray-500 mt-1">{alert.timestamp}</p>
                  </div>
                  <span className={"px-2 py-1 text-xs font-medium rounded-full ".concat(alert.severity === 'high' ? 'bg-red-100 text-red-800' :
                alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800')}>
                    {alert.severity}
                  </span>
                </div>); })}
            </div>
          </div>
        </div>

        {/* Model Health Summary */}
        <div className="bg-white rounded-lg shadow-sm border p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-6">Model Health Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
                <lucide_react_1.CheckCircle className="w-8 h-8 text-green-600"/>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Model Performance</h3>
              <p className="text-gray-600">All metrics within acceptable thresholds</p>
            </div>
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-yellow-100 rounded-full mb-4">
                <lucide_react_1.AlertTriangle className="w-8 h-8 text-yellow-600"/>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Data Quality</h3>
              <p className="text-gray-600">Minor drift detected in debt_to_income feature</p>
            </div>
            <div className="text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 rounded-full mb-4">
                <lucide_react_1.Activity className="w-8 h-8 text-green-600"/>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">System Health</h3>
              <p className="text-gray-600">All systems operational and responsive</p>
            </div>
          </div>
        </div>
      </div>
    </div>);
};
exports.default = MortgageModelMonitoringDashboard;
