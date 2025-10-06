const mongoose = require('mongoose');

const pupilMeasurementSchema = new mongoose.Schema({
  pd: {
    type: Number,
    required: true
  },
  leftNose: {
    type: Number,
    required: true
  },
  rightNose: {
    type: Number,
    required: true
  },
  tilt: {
    type: Number,
    required: true
  },
  distance: {
    type: Number,
    default: null
  },
  accuracy: {
    type: String,
    enum: ['High', 'Medium'],
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
  timestamps: true // Adds createdAt and updatedAt
});

module.exports = mongoose.model('PupilMeasurement', pupilMeasurementSchema);