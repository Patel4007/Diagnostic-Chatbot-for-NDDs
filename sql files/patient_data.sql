-- phpMyAdmin SQL Dump
-- version 5.1.3
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1:3306
-- Generation Time: Apr 26, 2022 at 04:35 PM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 8.1.5

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

--
-- Database: `rasa_input`
--

-- --------------------------------------------------------

--
-- Table structure for table `patient_data`
--

CREATE TABLE `patient_data` (
  `id` int(11) NOT NULL,
  `email` varchar(50) NOT NULL,
  `personName` varchar(255) NOT NULL,
  `summary` varchar(255) NOT NULL,
  `motor_skills` varchar(255) NOT NULL,
  `eye_hand_coord` varchar(255) NOT NULL,
  `compulsive` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `patient_data`
--

INSERT INTO `patient_data` (`id`, `email`, `personName`, `summary`, `motor_skills`, `eye_hand_coord`, `compulsive`) VALUES
(1, 'patel41100@gmail.com', 'Drishti Patel', 'She has gait issues, does repetitive motions and doctor said that she might have autism.', 'fail', 'fail', 'fail'),
(2, 'njoshi@gmail.com', 'rahul joshi', 'he seems inattentive most of the time and obsessed with following fixed routines that often seems abnormal.', 'pass', 'fail', 'fail'),
(3, 'hirak.dalwadi@gmail.com', 'tirth ', 'my kid flaps his hands constantly and does not show facial expressions at all.', 'fail', 'pass', 'fail');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `patient_data`
--
ALTER TABLE `patient_data`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `patient_data`
--
ALTER TABLE `patient_data`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;
COMMIT;
