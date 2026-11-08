// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract FederatedModelRegistry {
    address public immutable owner;
    uint256 public latestRound;
    mapping(uint256 => bytes32) public modelHashes;

    event ModelRecorded(
        uint256 indexed round,
        bytes32 indexed modelHash,
        uint256 timestamp,
        address indexed submitter
    );

    error Unauthorized();
    error InvalidRound();

    constructor() {
        owner = msg.sender;
    }

    function recordModel(uint256 round, bytes32 modelHash) external {
        if (msg.sender != owner) revert Unauthorized();
        if (round != latestRound + 1 || modelHash == bytes32(0)) revert InvalidRound();

        latestRound = round;
        modelHashes[round] = modelHash;
        emit ModelRecorded(round, modelHash, block.timestamp, msg.sender);
    }
}
