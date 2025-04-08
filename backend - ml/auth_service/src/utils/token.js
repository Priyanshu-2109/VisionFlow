const jwt = require("jsonwebtoken");

exports.generateToken = (userId, expiresIn = "15m") => {
  return jwt.sign({ userId }, process.env.JWT_SECRET, { expiresIn });
};

exports.generateRefreshToken = (userId) => {
  return jwt.sign({ userId }, process.env.JWT_REFRESH_SECRET, {
    expiresIn: "7d",
  });
};
