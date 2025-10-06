const mongoose = require('mongoose');

const frameMeasurementSchema = new mongoose.Schema({
  leftWidth: {
    type: Number,
    required: true
  },
  leftHeight: {
    type: Number,
    required: true
  },
  rightWidth: {
    type: Number,
    required: true
  },
  rightHeight: {
    type: Number,
    required: true
  },
  bridge: {
    type: Number,
    required: true
  },
  filename: {
    type: String,
    required: true
  },
  image: {
    type: String, // Base64 string
    required: true
  },
  timestamp: {
    type: Date,
    required: true
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('FrameMeasurement', frameMeasurementSchema);